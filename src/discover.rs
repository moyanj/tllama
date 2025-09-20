use lazy_static::lazy_static;
use serde_json::Value;
use std::collections::HashSet;
use std::env;
use std::fs;
use std::io::Read;
use std::path::Path;
use std::path::PathBuf;
use std::sync::Mutex;
use walkdir::WalkDir;

lazy_static! {
    pub static ref MODEL_DISCOVERER: Mutex<ModelDiscover> = Mutex::new(ModelDiscover::new());
}

#[derive(Debug, Clone)]
pub enum ModelType {
    Gguf,
    Safetensors,
}
#[derive(Debug, Clone)]
pub struct Model {
    pub model_type: ModelType,
    pub model_path: PathBuf,
    pub model_name: String,
    pub model_size: u64,
    pub model_template: Option<String>,
}
pub struct ModelDiscover {
    model_list: Vec<Model>,
    scan_all_paths: bool,
}

impl ModelDiscover {
    pub fn new() -> Self {
        ModelDiscover {
            model_list: Vec::new(),
            scan_all_paths: false,
        }
    }

    pub fn scan_all_paths(&mut self) {
        self.scan_all_paths = true;
    }

    /// 核心方法：扫描所有已知路径并填充模型列表。
    pub fn discover(&mut self) {
        self.model_list.clear();
        let search_paths = self.make_search_paths(true);
        for path in search_paths {
            if path.join("blobs").is_dir() && path.join("manifests").is_dir() {
                // Ollama 模型目录
                self.discover_ollama_models(&path.as_path());
                continue;
            }
            for entry in WalkDir::new(&path)
                .into_iter()
                .filter_map(Result::ok)
                .filter(|e| e.file_type().is_file())
            {
                println!("Scanning file: {:?}", entry.path());
                let full_path = entry.path();
                if self.check_exclude(&full_path) {
                    continue;
                }

                match full_path.metadata() {
                    Ok(meta) => {
                        if meta.len() < 50 * 1024 * 1024 {
                            // 文件小于 50MB，跳过
                            continue;
                        }
                    }
                    Err(_) => continue,
                }
                println!("Checking file: {:?}", full_path);
                if self.check_gguf_format(&full_path) {
                    self.model_list.push(Model {
                        model_name: full_path.file_stem().unwrap().to_string_lossy().to_string(),
                        model_type: ModelType::Gguf,
                        model_path: path.to_path_buf(),
                        model_size: full_path.metadata().unwrap().len(),
                        model_template: None,
                    });
                } else if self.check_safetensors_format(&full_path) {
                    self.model_list.push(Model {
                        model_name: full_path.file_stem().unwrap().to_string_lossy().to_string(),
                        model_type: ModelType::Safetensors,
                        model_path: path.to_path_buf(),
                        model_size: full_path.metadata().unwrap().len(),
                        model_template: None,
                    });
                } else {
                    continue;
                }
            }
        }
    }

    fn discover_ollama_models(&mut self, path: &Path) {
        let manifests_path = path.join("manifests");
        if !manifests_path.is_dir() {
            return;
        }
        for entry in WalkDir::new(&manifests_path)
            .into_iter()
            .filter_map(Result::ok)
            .filter(|e| e.file_type().is_file())
        {
            let full_path = entry.path();
            let file_rel_path = match full_path.strip_prefix(&manifests_path) {
                Ok(p) => p,
                Err(_) => continue,
            };

            let mut components: Vec<&str> = file_rel_path
                .components()
                .filter_map(|c| c.as_os_str().to_str())
                .collect();
            if components.is_empty() {
                continue;
            }
            // [新逻辑] 检查域名部分
            let domain = components[0];
            if domain == "registry.ollama.ai" {
                // 仅当域名是官方注册表时，我们才简化名称
                components.remove(0); // 移除 "registry.ollama.ai"
                if !components.is_empty() && components[0] == "library" {
                    components.remove(0); // 移除 "library"
                }
            }
            // 对于所有其他域名 (e.g., "localhost", "my-registry.com")，
            // 我们保留完整的路径来避免命名冲突，所以不做任何操作。

            if components.len() < 2 {
                // 至少需要 model_name 和 tag
                continue;
            }
            // 将最后一部分（标签）与前面的部分（模型名）用 ':' 连接
            let tag = components.pop().unwrap(); // 安全的 unwrap，因为已检查 len >= 2
            let model_repo = components.join("/");
            let model_name = format!("{}:{}", model_repo, tag);

            let json_content = match std::fs::read_to_string(full_path) {
                Ok(json) => json,
                Err(_) => continue,
            };
            let manifest: Value = match serde_json::from_str(&json_content) {
                Ok(v) => v,
                Err(_) => continue,
            };

            let model_size: u64 = manifest["layers"].as_array().map_or(0, |layers| {
                layers
                    .iter()
                    .filter_map(|layer| layer["size"].as_i64())
                    .sum()
            }) as u64;
            if model_size == 0 {
                continue;
            }
            let model_template: Option<String> = manifest["layers"] // <-- 修正拼写
                .as_array()
                .and_then(|layers| {
                    // 1. 找到包含模板信息的 layer
                    layers
                        .iter()
                        .find(|layer| layer["mediaType"] == "application/vnd.ollama.image.template")
                })
                .and_then(|template_layer| {
                    // 2. 从该 layer 中获取 digest (e.g., "sha256:abcdef...")
                    template_layer["digest"].as_str()
                })
                .and_then(|digest| {
                    // 3. 将 digest 转换为 blob 文件名 (e.g., "sha256-abcdef...")
                    let blob_filename = digest.replace(':', "-");
                    let blob_path = path.join("blobs").join(blob_filename);

                    // 4. 读取 blob 文件的内容，这正是模板字符串
                    fs::read_to_string(blob_path).ok()
                });

            let model_path = manifest["layers"]
                .as_array()
                .and_then(|layers| {
                    // 1. 找到包含模板信息的 layer
                    layers
                        .iter()
                        .find(|layer| layer["mediaType"] == "application/vnd.ollama.image.model")
                })
                .and_then(|template_layer| {
                    // 2. 从该 layer 中获取 digest (e.g., "sha256:abcdef...")
                    template_layer["digest"].as_str()
                })
                .and_then(|digest| {
                    // 3. 将 digest 转换为 blob 文件名 (e.g., "sha256-abcdef...")
                    let blob_filename = digest.replace(':', "-");
                    let p = path.join("blobs").join(blob_filename);
                    if !p.exists() {
                        return None;
                    }
                    Some(p.to_path_buf())
                });
            let model_path = match model_path {
                Some(p) => p,
                None => continue,
            };

            let model = Model {
                model_type: ModelType::Gguf,
                model_path,
                model_name,
                model_size,
                model_template,
            };
            self.model_list.push(model);
        }
    }

    fn check_gguf_format(&self, path: &Path) -> bool {
        if let Ok(mut file) = fs::File::open(path) {
            let mut magic = [0u8; 4];
            if let Ok(_) = file.read_exact(&mut magic) {
                return &magic == b"GGUF";
            }
        }
        false
    }

    fn check_safetensors_format(&self, path: &Path) -> bool {
        if let Ok(mut file) = fs::File::open(path) {
            // 读取元数据长度
            let mut len_bytes = [0u8; 8];
            if let Ok(_) = file.read_exact(&mut len_bytes) {
                let len = u64::from_le_bytes(len_bytes) as usize;
                if len > 50 * 1024 * 1024 {
                    // 元数据长度不应超过 50MB
                    return false;
                }
                // 读取元数据
                let mut json_bytes = vec![0u8; len];
                if let Ok(_) = file.read_exact(&mut json_bytes) {
                    if let Ok(json_str) = String::from_utf8(json_bytes) {
                        if let Ok(_) = serde_json::from_str::<Value>(&json_str) {
                            // 检查是否包含 "metadata" 字段
                            return true;
                        }
                    }
                }
            }
        }
        false
    }

    /// 构建一个包含所有潜在模型目录的列表
    pub fn make_search_paths(&self, check_existence: bool) -> Vec<PathBuf> {
        if self.scan_all_paths {
            #[cfg(unix)]
            {
                return vec![PathBuf::from("/")];
            }
            #[cfg(windows)]
            {
                // Windows 下扫描所有驱动器
                let mut drives = Vec::new();
                for letter in b'A'..=b'Z' {
                    let drive = format!("{}:\\", letter as char);
                    if Path::new(&drive).exists() {
                        drives.push(PathBuf::from(drive));
                    }
                }
                return drives;
            }
        }

        let mut paths = HashSet::new();
        if let Ok(rllama_paths_str) = env::var("RLLAMA_MODEL_PATHS") {
            for path_str in rllama_paths_str.split(',') {
                let trimmed_path = path_str.trim();
                if !trimmed_path.is_empty() {
                    paths.insert(PathBuf::from(trimmed_path));
                }
            }
        }
        paths.insert(PathBuf::from("./models"));
        let home_dir = dirs::home_dir();
        let cache_dir = dirs::cache_dir();
        if let Some(home) = home_dir {
            paths.insert(home.join("Downloads"));
            paths.insert(home.join("Documents").join("models"));
            paths.insert(home.join("jan").join("models"));
            #[cfg(feature = "engine-llama-cpp")]
            if cfg!(target_os = "macos") || cfg!(target_os = "linux") {
                paths.insert(home.join(".ollama").join("models"));
            }
        }
        if let Some(cache) = cache_dir {
            #[cfg(feature = "engine-llama-cpp")]
            {
                paths.insert(cache.join("lm-studio").join("models"));
            }
            paths.insert(cache.join("gpt4all"));
            #[cfg(feature = "engine-hf")]
            {
                let hf_home = env::var("HF_HOME")
                    .ok()
                    .map(PathBuf::from)
                    .unwrap_or_else(|| cache.join("huggingface"));
                paths.insert(hf_home.join("hub"));
            }
        }
        let final_paths: Vec<PathBuf> = paths.into_iter().collect();
        if check_existence {
            final_paths.into_iter().filter(|p| p.is_dir()).collect()
        } else {
            final_paths
        }
    }

    fn check_exclude(&self, path: &Path) -> bool {
        if !self.scan_all_paths {
            return false;
        }
        let uni_exclude_list = vec![".git", "node_modules", "venv", "__pycache__"];
        #[cfg(target_os = "linux")]
        let exclude_path = vec![
            "/var", "/proc", "/run", "/sys", "/dev", "/lib", "/lib64", "/snap", "/boot",
        ];
        #[cfg(not(target_os = "linux"))]
        let exclude_path = vec![
            "C:\\Windows",
            "C:\\Program Files",
            "C:\\Program Files (x86)",
            "C:\\ProgramData",
        ];
        if path
            .components()
            .filter_map(|c| c.as_os_str().to_str()) // 转换为 &str
            .any(|c_str| uni_exclude_list.contains(&c_str.to_lowercase().as_str()))
        {
            return true;
        }

        for excl in exclude_path {
            if path.starts_with(excl) {
                return true;
            }
        }

        false
    }

    /// 获取发现的模型列表的只读引用 (无变化)
    pub fn get_model_list(&self) -> &Vec<Model> {
        &self.model_list
    }
}
