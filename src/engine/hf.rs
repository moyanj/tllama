use std::collections::HashMap;
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::thread;

use super::EngineBackend;
use crate::{discover::Model, engine::EngineConfig};
use anyhow::Result;
use lazy_static::lazy_static;
use serde_json::Value;
use serde_json::json;
use tempfile::NamedTempFile;
use uuid::Uuid;

type Sender = Box<dyn FnMut(Value) + Send>;

lazy_static! {
    static ref PYTHON_BACKEND: Mutex<PythonBackend> = Mutex::new(PythonBackend::new().unwrap());
}

struct PythonBackend {
    stdin: Arc<Mutex<ChildStdin>>,
    response_senders: Arc<Mutex<HashMap<String, Sender>>>,
    _child: Child, // 必须保留 Child，防止子进程被提前终止
}
impl PythonBackend {
    pub fn new() -> Result<Self> {
        // 创建临时脚本文件
        let mut tmpfile = NamedTempFile::new()?;
        write!(tmpfile, "{}", include_str!("../assets/daemon.py"))?;
        let script_path = tmpfile.into_temp_path(); // 保持临时文件存活

        // 启动子进程
        let mut child = Command::new("python")
            .arg(&script_path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()?;

        // 获取 stdin 和 stdout
        let stdin = Arc::new(Mutex::new(child.stdin.take().unwrap()));
        let stdout = child.stdout.take().unwrap();

        // 共享的 response_senders
        let response_senders: Arc<Mutex<HashMap<String, Sender>>> =
            Arc::new(Mutex::new(HashMap::new()));

        // 共享 child 句柄
        let child_arc = Arc::new(Mutex::new(child));

        // 捕获 response_senders 用于读取线程
        let response_senders_clone = Arc::clone(&response_senders);

        thread::spawn(move || {
            let reader = BufReader::new(stdout);
            for line in reader.lines() {
                let line = match line {
                    Ok(l) => l,
                    Err(_) => continue,
                };

                match serde_json::from_str::<Value>(&line) {
                    Ok(json) => {
                        if let Some(id) = json["req_id"].as_str() {
                            if json.get("token").is_some() && json.get("done").is_some() {
                                let mut senders = response_senders_clone.lock().unwrap();
                                if let Some(sender) = senders.get_mut(id) {
                                    sender(json);
                                }
                            }
                        }
                    }
                    Err(e) => {
                        eprintln!("[PythonBackend] JSON 解析失败: {}", e);
                    }
                }
            }
            eprintln!("[PythonBackend] stdout 流结束，读取线程退出");
        });

        // 单独线程等待子进程结束（避免阻塞）
        let response_senders_for_wait = Arc::clone(&response_senders);
        let child_arc_for_wait = Arc::clone(&child_arc);

        thread::spawn(move || {
            let mut child = child_arc_for_wait.lock().unwrap();
            let status = child.wait().unwrap();
            eprintln!("[Python] 进程已退出，状态: {}", status);
            // 可选：在此通知所有 sender 子进程已断开
            let mut senders = response_senders_for_wait.lock().unwrap();
            senders.clear(); // 避免后续调用无效回调
        });

        Ok(PythonBackend {
            stdin,
            response_senders,
            _child: Arc::try_unwrap(child_arc).unwrap().into_inner().unwrap(), // 保存 child 以维持进程存活
        })
    }

    // 可选：注册回调
    pub fn register_callback<F>(&self, req_id: String, callback: F)
    where
        F: FnMut(Value) + Send + 'static,
    {
        let mut senders = self.response_senders.lock().unwrap();
        senders.insert(req_id, Box::new(callback));
    }

    // 可选：发送消息到 Python
    pub fn infer(&self, model_name: &str, prompt: &str, args: &EngineConfig) -> Result<String> {
        let req_id = Uuid::new_v4().to_string();
        let data = json!({
            "req_id": req_id.clone(),
            "model": model_name,
            "prompt": prompt,
            "args": args,
        });
        let mut stdin = self.stdin.lock().unwrap();
        writeln!(stdin, "{}", data)?;
        stdin.flush()?;
        Ok(req_id)
    }
}

pub struct TransformersEngine {
    model_info: Model,
    args: EngineConfig,
}

impl EngineBackend for TransformersEngine {
    fn new(args: &EngineConfig, model_info: &Model) -> Result<Self> {
        let engine = TransformersEngine {
            model_info: model_info.clone(),
            args: args.clone(),
        };
        Ok(engine)
    }
    fn infer(
        &self,
        prompt: &str,
        option: Option<&EngineConfig>,
        callback: Option<Box<dyn FnMut(String) + Send>>,
    ) -> Result<String> {
        let args = option.unwrap_or(&self.args);
        let req_id = PYTHON_BACKEND
            .lock()
            .map_err(|e| anyhow::anyhow!(e.to_string()))?
            .infer(self.model_info.model_path.to_str().unwrap(), prompt, args)?;
        let mut callback = callback;
        PYTHON_BACKEND
            .lock()
            .map_err(|e| anyhow::anyhow!(e.to_string()))?
            .register_callback(req_id, move |token| {
                if let Some(cb) = callback.as_mut() {
                    cb(token["token"].as_str().unwrap().to_string());
                }
            });
        Ok("1".to_string())
    }
    fn get_model_info(&self) -> Model {
        self.model_info.clone()
    }
}
