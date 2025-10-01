use super::EngineBackend;
use crate::engine::EngineCallback;
use crate::{discover::Model, engine::EngineConfig};
use anyhow::Result;
use lazy_static::lazy_static;
use serde_json::Value;
use serde_json::json;
use std::collections::HashMap;
use std::io::BufRead;
use std::io::{BufReader, Write};
use std::process::{ChildStdin, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::thread;
use tempfile::NamedTempFile;
use uuid::Uuid;

// ========== 回调类型定义 ==========
type ResponseCallback = Box<dyn FnMut(Value) + Send>;

// ========== 全局单例 Python Backend ==========
lazy_static! {
    pub static ref PYTHON_BACKEND: Mutex<PythonBackend> = {
        match PythonBackend::new() {
            Ok(backend) => Mutex::new(backend),
            Err(e) => {
                eprintln!("[FATAL] Can't start Python backend:");
                eprintln!("错误: {}", e);
                panic!();
            }
        }
    };
}

// ========== PythonBackend 结构体 ==========
pub struct PythonBackend {
    stdin: Arc<Mutex<ChildStdin>>,
    response_senders: Arc<Mutex<HashMap<String, ResponseCallback>>>,
}

impl PythonBackend {
    pub fn new() -> Result<Self> {
        // 创建临时脚本文件
        let mut tmpfile = NamedTempFile::new()?;
        write!(tmpfile, "{}", include_str!("../assets/hf_daemon.py"))?;
        let (_file, path) = tmpfile.keep()?;

        // 启动 Python 子进程，同时捕获 stderr
        let mut child = Command::new("python")
            .arg(&path)
            .env("TLLAMA_DAEMON", "1")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| {
                anyhow::anyhow!("Failed to start Python process: {}. Make sure Python is installed and in PATH.", e)
            })?;

        let stdin = Arc::new(Mutex::new(child.stdin.take().unwrap()));
        let stdout = child.stdout.take().unwrap();
        let stderr = child.stderr.take().unwrap(); // 获取 stderr

        // 启动 stderr 读取线程：实时输出 Python 错误信息
        thread::spawn(move || {
            let reader = BufReader::new(stderr);
            for line in reader.lines() {
                match line {
                    Ok(line) => eprintln!("[Python STDERR] {}", line),
                    Err(e) => eprintln!("[PythonBackend] Can't read stderr: {}", e),
                }
            }
        });

        // 共享的回调映射表
        let response_senders: Arc<Mutex<HashMap<String, Box<dyn FnMut(Value) + Send + 'static>>>> =
            Arc::new(Mutex::new(HashMap::new()));
        let response_senders_clone = Arc::clone(&response_senders);

        // 启动读取线程：监听 Python 输出
        thread::spawn(move || {
            let reader = BufReader::new(stdout);
            for line in reader.lines() {
                let line = match line {
                    Ok(l) => l,
                    Err(e) => {
                        eprintln!("[PythonBackend] 读取 stdout 失败: {}", e);
                        continue;
                    }
                };

                match serde_json::from_str::<Value>(&line) {
                    Ok(json) => {
                        if let Some(id) = json["req_id"].as_str() {
                            let mut senders = match response_senders_clone.lock() {
                                Ok(guard) => guard,
                                Err(_) => {
                                    eprintln!("[PythonBackend] 回调锁被污染");
                                    return;
                                }
                            };
                            if let Some(sender) = senders.get_mut(id) {
                                sender(json.clone());
                            }
                            // 如果是结束消息，清理回调
                            if json.get("done").is_some() {
                                senders.remove(id);
                            }
                        }
                    }
                    Err(e) => {
                        eprintln!("[PythonBackend] JSON Parse Fault: {}: {}", e, line);
                    }
                }
            }
        });

        // 启动等待线程：监控子进程退出
        let response_senders_for_wait = Arc::clone(&response_senders);
        thread::spawn(move || {
            let status = match child.wait() {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("[PythonBackend] 等待子进程失败: {}", e);
                    return;
                }
            };

            if !status.success() {
                eprintln!("[PythonBackend] Python 进程异常退出，状态: {}", status);
                std::process::exit(1);
            } else {
                eprintln!("[PythonBackend] Python 进程正常退出");
            }

            // 清理所有未完成的回调
            let mut senders = match response_senders_for_wait.lock() {
                Ok(guard) => guard,
                Err(_) => return,
            };
            senders.clear();
        });

        Ok(PythonBackend {
            stdin,
            response_senders,
        })
    }

    /// 发送推理请求并注册响应回调
    pub fn infer_with_callback<F>(
        &self,
        model_name: &str,
        prompt: &str,
        args: &EngineConfig,
        callback: F,
    ) -> Result<String>
    where
        F: FnMut(Value) + Send + 'static,
    {
        let req_id = Uuid::new_v4().to_string();

        // 注册回调
        {
            let mut senders = self
                .response_senders
                .lock()
                .map_err(|e| anyhow::anyhow!("锁冲突: {:?}", e))?;
            senders.insert(req_id.clone(), Box::new(callback));
        }

        // 构造请求
        let request = json!({
            "req_id": req_id,
            "model": model_name,
            "prompt": prompt,
            "args": args,
        });

        // 发送请求
        {
            let mut stdin = self
                .stdin
                .lock()
                .map_err(|e| anyhow::anyhow!("stdin 锁失败: {:?}", e))?;
            writeln!(stdin, "{}", serde_json::to_string(&request)?)?;
            stdin.flush()?; // 关键：必须 flush
        }

        Ok(req_id)
    }

    pub fn load_model(&self, model: &str) -> Result<()> {
        let req_id = Uuid::new_v4().to_string();
        let request = json!({
            "req_id": req_id,
            "cmd": "load",
            "model": model,
        });

        // 创建同步信号
        let loaded = Arc::new(Mutex::new(false));
        let loaded_clone = Arc::clone(&loaded);

        // 注册临时回调，等待加载完成
        {
            let mut senders = self
                .response_senders
                .lock()
                .map_err(|e| anyhow::anyhow!("锁冲突: {:?}", e))?;
            senders.insert(
                req_id.clone(),
                Box::new(move |json: Value| {
                    if json.get("loaded").is_some() || json.get("error").is_some() {
                        let mut loaded = loaded_clone.lock().unwrap();
                        *loaded = true;
                    }
                }),
            );
        }

        // 发送请求
        {
            let mut stdin = self
                .stdin
                .lock()
                .map_err(|e| anyhow::anyhow!("stdin 锁失败: {:?}", e))?;
            writeln!(stdin, "{}", serde_json::to_string(&request)?)?;
            stdin.flush()?; // 关键：必须 flush
        }

        // 等待加载完成
        loop {
            thread::sleep(std::time::Duration::from_millis(10));
            let loaded = loaded.lock().unwrap();
            if *loaded {
                break;
            }
        }

        Ok(())
    }

    pub fn unload_model(&self, model: &str) -> Result<()> {
        let req_id = Uuid::new_v4().to_string();
        let request = json!({
            "req_id": req_id,
            "cmd": "unload",
            "model": model,
        });

        // 创建同步信号
        let unloaded = Arc::new(Mutex::new(false));
        let unloaded_clone = Arc::clone(&unloaded);

        // 注册临时回调，等待卸载完成
        {
            let mut senders = self
                .response_senders
                .lock()
                .map_err(|e| anyhow::anyhow!("锁冲突: {:?}", e))?;
            senders.insert(
                req_id.clone(),
                Box::new(move |json: Value| {
                    if json.get("unloaded").is_some() || json.get("error").is_some() {
                        let mut unloaded = unloaded_clone.lock().unwrap();
                        *unloaded = true;
                    }
                }),
            );
        }

        // 发送请求
        {
            let mut stdin = self
                .stdin
                .lock()
                .map_err(|e| anyhow::anyhow!("stdin 锁失败: {:?}", e))?;
            writeln!(stdin, "{}", serde_json::to_string(&request)?)?;
            stdin.flush()?; // 关键：必须 flush
        }

        // 等待卸载完成
        loop {
            thread::sleep(std::time::Duration::from_millis(10));
            let unloaded = unloaded.lock().unwrap();
            if *unloaded {
                break;
            }
        }

        Ok(())
    }
}

impl Drop for PythonBackend {
    fn drop(&mut self) {
        let request = json!({
            "req_id": "__exit__",
            "cmd": "exit",
        });
        {
            let mut stdin = self
                .stdin
                .lock()
                .map_err(|e| anyhow::anyhow!("stdin 锁失败: {:?}", e))
                .unwrap();
            writeln!(stdin, "{}", serde_json::to_string(&request).unwrap()).unwrap();
            let _ = stdin.flush(); // 关键：必须 flush
        }
    }
}

// ========== TransformersEngine 实现 ==========
pub struct TransformersEngine {
    model_info: Model,
    args: EngineConfig,
}

impl EngineBackend for TransformersEngine {
    fn new(args: &EngineConfig, model_info: &Model) -> Result<Self> {
        let backend = PYTHON_BACKEND.lock().expect("锁被污染");
        backend.load_model(model_info.path.to_str().unwrap())?;
        Ok(Self {
            model_info: model_info.clone(),
            args: args.clone(),
        })
    }

    fn infer(
        &self,
        prompt: &str,
        option: Option<&EngineConfig>,
        callback: Option<EngineCallback>,
    ) -> Result<String> {
        let args = option.unwrap_or(&self.args);
        let model_path = self
            .model_info
            .path
            .to_str()
            .ok_or_else(|| anyhow::anyhow!("模型路径包含非 UTF-8 字符"))?;

        // 获取全局 backend
        let backend = PYTHON_BACKEND
            .lock()
            .map_err(|e| anyhow::anyhow!("PythonBackend 锁被污染: {:?}", e))?;

        // 创建同步信号
        let finished = Arc::new(Mutex::new(false));
        let finished_clone = Arc::clone(&finished);

        // 将 callback 包装为 Arc<Mutex<Option<...>>>，以便在闭包中多次使用
        let shared_callback: Arc<Mutex<Option<EngineCallback>>> = Arc::new(Mutex::new(callback));

        // 创建闭包，适配 PythonBackend 的 FnMut(Value) 接口
        let closure_callback = {
            let shared_callback = Arc::clone(&shared_callback);
            let finished_clone = Arc::clone(&finished_clone);
            move |json: Value| {
                // 检查是否完成
                if json.get("done").is_some() || json.get("error").is_some() {
                    let mut finished = finished_clone.lock().unwrap();
                    *finished = true;
                    return;
                }

                let token = json["token"].as_str().unwrap_or_default();
                let mut guard = shared_callback.lock().unwrap();
                if let Some(ref mut cb) = *guard {
                    cb(token.to_string());
                }
            }
        };

        // 发送请求并注册回调
        let req_id = backend.infer_with_callback(model_path, prompt, args, closure_callback)?;

        // 等待生成完成
        loop {
            thread::sleep(std::time::Duration::from_millis(10));
            let finished = finished.lock().unwrap();
            if *finished {
                break;
            }
        }

        Ok(req_id)
    }

    fn get_model_info(&self) -> Model {
        self.model_info.clone()
    }
}

impl Drop for TransformersEngine {
    fn drop(&mut self) {
        let backend = PYTHON_BACKEND
            .lock()
            .map_err(|e| anyhow::anyhow!("PythonBackend 锁被污染: {:?}", e))
            .unwrap();
        let _ = backend.unload_model(self.model_info.path.to_str().unwrap());
    }
}
