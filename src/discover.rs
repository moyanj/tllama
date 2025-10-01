use glob::Pattern;
use lazy_static::lazy_static;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashSet;
use std::env;
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use walkdir::WalkDir;

lazy_static! {
    pub static ref MODEL_DISCOVERER: Mutex<ModelDiscover> = Mutex::new({
        let mut discoverer = ModelDiscover::new();
        discoverer.discover();
        discoverer
    });
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModelType {
    Gguf,
    Transformers,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Model {
    pub format: ModelType,
    pub path: PathBuf,
    pub name: String,
    pub size: u64,
    pub template: Option<String>,
}

impl Model {
    pub fn from_path(path: &String) -> Self {
        Model {
            path: PathBuf::from(path),
            name: path.to_string(),
            format: ModelType::Gguf,
            size: 0,
            template: None,
        }
    }
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

    pub fn scan_all_paths(&mut self, scan: bool) {
        self.scan_all_paths = scan;
    }
    /// core method, scan model directory
    pub fn discover(&mut self) {
        self.model_list.clear();
        let search_paths = self.make_search_paths(true);
        for path in search_paths {
            if directory_has_features(&path, &["manifests", "blobs", "blobs/sha256-*"]) {
                // Ollama Models
                self.discover_ollama_models(&path.as_path());
                continue;
            }
            if directory_has_features(&path, &["*/blobs", "*/refs", "*/snapshots"]) {
                // HuggingFace Cached Models
                self.discover_hf_models(&path);
                continue;
            }
            for entry in WalkDir::new(&path)
                .into_iter()
                .filter_map(Result::ok)
                .filter(|e| e.file_type().is_file())
            {
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
                if self.check_gguf_format(&full_path) {
                    self.model_list.push(Model {
                        name: full_path.file_stem().unwrap().to_string_lossy().to_string(),
                        format: ModelType::Gguf,
                        path: path.to_path_buf(),
                        size: full_path.metadata().unwrap().len(),
                        template: None,
                    });
                } else if self.check_safetensors_format(&full_path) {
                    self.model_list.push(Model {
                        name: full_path.file_stem().unwrap().to_string_lossy().to_string(),
                        format: ModelType::Transformers,
                        path: path.to_path_buf(),
                        size: full_path.metadata().unwrap().len(),
                        template: None,
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
                format: ModelType::Gguf,
                path: model_path,
                name: model_name,
                size: model_size,
                template: model_template,
            };
            self.model_list.push(model);
        }
    }

    fn discover_hf_models(&mut self, path: &Path) {
        for model in path.read_dir().expect("Failed to read directory") {
            if let Ok(entry) = model {
                let model_dir = entry.path();
                if !entry.file_type().map_or(false, |ft| ft.is_dir())
                    || !entry.file_name().to_string_lossy().starts_with("models--")
                {
                    continue;
                }

                // 检查是否包含必要的文件
                if !directory_has_features(
                    &model_dir,
                    &[
                        "snapshots/*/config.json",
                        "snapshots/*/tokenizer_config.json",
                    ],
                ) {
                    continue;
                }

                // 解析模型名称：models--owner--repo -> owner/repo
                let file_name = entry.file_name();
                let file_name_str = file_name.to_string_lossy();
                let stripped = &file_name_str["models--".len()..];
                let parts: Vec<&str> = stripped.splitn(2, "--").collect();
                let model_name = if parts.len() == 2 {
                    format!("{}/{}", parts[0], parts[1].replace("--", "/"))
                } else {
                    stripped.replace("--", "/")
                };

                // 查找 snapshot 目录下的所有快照（通常只有一个，但支持多个）
                let snapshot_path = model_dir.join("snapshots");
                if !snapshot_path.is_dir() {
                    continue;
                }

                for snapshot in snapshot_path.read_dir().expect("Failed to read snapshots") {
                    if let Ok(snapshot_entry) = snapshot {
                        if !snapshot_entry.file_type().map_or(false, |ft| ft.is_dir()) {
                            continue;
                        }

                        let snapshot_dir = snapshot_entry.path();
                        let tokenizer_config_path = snapshot_dir.join("tokenizer_config.json");

                        // 读取 chat template
                        let chat_template = if tokenizer_config_path.exists() {
                            if let Ok(content) = fs::read_to_string(&tokenizer_config_path) {
                                if let Ok(json) = serde_json::from_str::<Value>(&content) {
                                    json["chat_template"]
                                        .as_str()
                                        .map(|s| s.to_string())
                                        .or_else(|| {
                                            // 回退到特殊字段如 tokenizer.chat_template（罕见情况）
                                            json.get("tokenizer")
                                                .and_then(|t| t["chat_template"].as_str())
                                                .map(|s| s.to_string())
                                        })
                                } else {
                                    None
                                }
                            } else {
                                None
                            }
                        } else {
                            None
                        };

                        // 回退到默认模板
                        let effective_template = chat_template
                            .unwrap_or_else(|| crate::template::get_default_template());

                        // 统计模型文件总大小
                        let mut total_size: u64 = 0;
                        let mut file_count = 0;
                        for entry in WalkDir::new(&snapshot_dir)
                            .into_iter()
                            .filter_map(Result::ok)
                            .filter(|e| e.file_type().is_file() || e.file_type().is_symlink())
                        {
                            if entry.file_type().is_symlink() {
                                // 解析symlink
                                let target = entry
                                    .path()
                                    .parent()
                                    .unwrap()
                                    .join(entry.path().read_link().unwrap());
                                let metadata = target.metadata().unwrap();
                                total_size += metadata.len();
                                file_count += 1;
                            } else if let Ok(metadata) = entry.metadata() {
                                total_size += metadata.len();
                                file_count += 1;
                            }
                        }

                        // 如果没有有效文件，跳过
                        if file_count == 0 || total_size < 50 * 1024 * 1024 {
                            continue;
                        }

                        // 创建模型条目
                        let model = Model {
                            format: ModelType::Transformers,
                            path: snapshot_dir.clone(), // 指向 snapshot 目录
                            name: model_name.clone(),
                            size: total_size,
                            template: Some(effective_template),
                        };

                        self.model_list.push(model);
                    }
                }
            }
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
        for path_str in &(*crate::env::TLLAMA_MODEL_PATHS) {
            let trimmed_path = path_str.trim();
            if !trimmed_path.is_empty() {
                paths.insert(PathBuf::from(trimmed_path));
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
            //#[cfg(feature = "engine-hf")]
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

    pub fn find_model(&self, model_name: &str) -> Result<Model, Box<dyn std::error::Error>> {
        for model in &self.model_list {
            if model.name == model_name {
                return Ok(model.clone());
            }
        }
        Err(format!("Model {} not found", model_name).into())
    }
}

/// 检测目录是否拥有指定的特征
///
/// # 参数
/// - `dir_path`: 要检测的目录路径
/// - `features`: 特征列表，支持通配符和相对路径
///
/// # 返回值
/// - 如果目录包含所有指定的特征返回 true，否则返回 false
pub fn directory_has_features<P: AsRef<Path>>(dir_path: P, features: &[&str]) -> bool {
    let dir_path = dir_path.as_ref();

    if !dir_path.exists() || !dir_path.is_dir() {
        return false;
    }

    // 检查每个特征
    for feature in features {
        if !check_single_feature(dir_path, feature) {
            return false;
        }
    }

    true
}

/// 检查单个特征
fn check_single_feature(dir_path: &Path, feature: &str) -> bool {
    // 如果特征包含路径分隔符，需要特殊处理
    if feature.contains('/') || feature.contains('\\') {
        check_path_feature(dir_path, feature)
    } else {
        // 简单文件名匹配
        check_simple_feature(dir_path, feature)
    }
}

/// 检查简单文件名特征（不包含路径）
fn check_simple_feature(dir_path: &Path, pattern: &str) -> bool {
    let compiled_pattern = match Pattern::new(pattern) {
        Ok(p) => p,
        Err(_) => return false,
    };

    // 读取目录直接匹配
    if let Ok(entries) = fs::read_dir(dir_path) {
        for entry in entries {
            if let Ok(entry) = entry {
                let file_name = entry.file_name();
                let file_name_str = file_name.to_string_lossy();

                if compiled_pattern.matches(&file_name_str) {
                    return true;
                }
            }
        }
    }

    false
}

/// 检查包含路径的特征（如 "cyv/*/a.json"）
fn check_path_feature(dir_path: &Path, feature_pattern: &str) -> bool {
    // 构建完整的glob模式
    let full_pattern = if feature_pattern.starts_with('/') || feature_pattern.starts_with('\\') {
        // 如果是绝对路径，直接使用
        feature_pattern.to_string()
    } else {
        // 相对路径，基于目标目录构建完整路径
        let dir_str = dir_path.to_string_lossy();
        let normalized_dir = if dir_str.ends_with('/') || dir_str.ends_with('\\') {
            dir_str.to_string()
        } else {
            format!("{}/", dir_str)
        };
        format!("{}{}", normalized_dir, feature_pattern)
    };

    // 使用glob进行匹配
    match glob::glob(&full_pattern) {
        Ok(paths) => {
            for path in paths {
                if let Ok(path) = path {
                    if path.exists() {
                        return true;
                    }
                }
            }
            false
        }
        Err(_) => false,
    }
}
