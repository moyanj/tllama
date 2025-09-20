use lazy_static::lazy_static;
use std::collections::HashSet;
use std::env;
use std::fs;
use std::path::PathBuf;
use std::sync::{Mutex, MutexGuard};
use walkdir::WalkDir;

// --- 1. 全局模型发现器实例 ---
// 使用 Mutex 包装，使其可以在多线程环境中安全地修改
lazy_static! {
    static ref MODEL_DISCOVERER: Mutex<ModelDiscover> = Mutex::new(ModelDiscover::new());
}

// --- 2. 数据结构定义 ---
#[derive(Debug, Clone)] // 添加 Clone trait
pub enum ModelType {
    Gguf,
    Safetensors,
}

#[derive(Debug, Clone)] // 添加 Clone trait
pub struct Model {
    model_type: ModelType,
    model_path: PathBuf,
    model_name: String,
    model_size: i64, // 单位: 字节
}

pub struct ModelDiscover {
    model_list: Vec<Model>,
}

// --- 3. 核心实现 ---
impl ModelDiscover {
    /// 创建一个新的、空的 ModelDiscover 实例。
    pub fn new() -> Self {
        ModelDiscover {
            model_list: Vec::new(),
        }
    }

    /// 核心方法：扫描所有已知路径并填充模型列表。
    ///
    /// 此方法会清空现有的模型列表，然后根据启用的 feature flags
    /// (`engine-llama-cpp` for GGUF, `engine-hf` for Safetensors)
    /// 来递归地搜索并填充模型。
    pub fn discover(&mut self) {
        self.model_list.clear();
        let search_paths = self.make_search_paths(true); // true: 只搜索实际存在的目录

        for path in search_paths {
            // 使用 walkdir 进行高效的递归遍历
            // filter_map(Result::ok) 会优雅地跳过因权限等问题无法访问的文件/目录
            for entry in WalkDir::new(path).into_iter().filter_map(Result::ok) {
                if !entry.file_type().is_file() {
                    continue; // 跳过目录，只处理文件
                }

                let file_path = entry.path();

                // 根据文件扩展名和启用的 feature flag 来决定是否添加模型
                match file_path.extension().and_then(|s| s.to_str()) {
                    // 仅当 engine-llama-cpp feature 启用时，才识别 .gguf 文件
                    #[cfg(feature = "engine-llama-cpp")]
                    Some("gguf") => self.add_model_if_valid(file_path, ModelType::Gguf),

                    // 仅当 engine-hf feature 启用时，才识别 .safetensors 文件
                    #[cfg(feature = "engine-hf")]
                    Some("safetensors") => {
                        self.add_model_if_valid(file_path, ModelType::Safetensors)
                    }

                    _ => { /* 忽略所有其他类型的文件 */ }
                }
            }
        }
    }

    /// 辅助方法：验证文件元数据后，创建 Model 结构体并添加到列表中。
    fn add_model_if_valid(&mut self, path: &std::path::Path, model_type: ModelType) {
        if let Ok(metadata) = fs::metadata(path) {
            let model_name = path
                .file_stem() // 获取不带扩展名的文件名 (e.g., "llama-7b" from "llama-7b.gguf")
                .and_then(|s| s.to_str())
                .unwrap_or("unknown_model")
                .to_string();

            let model = Model {
                model_type,
                model_path: path.to_path_buf(),
                model_name,
                model_size: metadata.len() as i64,
            };
            self.model_list.push(model);
        }
    }

    /// 构建一个包含所有潜在模型目录的列表，这是整个发现过程的起点。
    pub fn make_search_paths(&self, check_existence: bool) -> Vec<PathBuf> {
        let mut paths = HashSet::new();

        // 1. (最高优先级) 从 RLLAMA_MODEL_PATHS 环境变量读取自定义路径
        if let Ok(rllama_paths_str) = env::var("RLLAMA_MODEL_PATHS") {
            // 使用逗号 ',' 作为通用分隔符，并处理可能存在的空格
            for path_str in rllama_paths_str.split(',') {
                let trimmed_path = path_str.trim();
                if !trimmed_path.is_empty() {
                    paths.insert(PathBuf::from(trimmed_path));
                }
            }
        }

        // 2. 添加相对于当前工作目录的 `./models` 路径
        paths.insert(PathBuf::from("./models"));

        // 3. 添加基于用户主目录和缓存目录的常见路径
        let home_dir = dirs::home_dir();
        let cache_dir = dirs::cache_dir();

        if let Some(home) = home_dir {
            paths.insert(home.join("Downloads")); // 通用下载目录
            paths.insert(home.join("Documents").join("models")); // 通用文档/模型目录
            paths.insert(home.join("jan").join("models")); // Jan.ai 模型路径

            #[cfg(feature = "engine-llama-cpp")]
            if cfg!(target_os = "macos") || cfg!(target_os = "linux") {
                paths.insert(home.join(".ollama").join("models")); // Ollama 模型路径
            }
        }

        if let Some(cache) = cache_dir {
            #[cfg(feature = "engine-llama-cpp")]
            {
                paths.insert(cache.join("lm-studio").join("models")); // LM Studio 模型路径
            }
            paths.insert(cache.join("gpt4all")); // GPT4All 模型路径

            #[cfg(feature = "engine-hf")]
            {
                // 优先使用 HF_HOME 环境变量，否则回退到标准缓存路径
                let hf_home = env::var("HF_HOME")
                    .ok()
                    .map(PathBuf::from)
                    .unwrap_or_else(|| cache.join("huggingface"));
                paths.insert(hf_home.join("hub")); // Hugging Face Hub 缓存路径
            }
        }

        // 4. 最后，将 HashSet 转换为 Vec，并根据参数决定是否过滤掉不存在的路径
        let final_paths: Vec<PathBuf> = paths.into_iter().collect();
        if check_existence {
            final_paths.into_iter().filter(|p| p.is_dir()).collect()
        } else {
            final_paths
        }
    }

    /// 获取发现的模型列表的只读引用。
    pub fn get_model_list(&self) -> &Vec<Model> {
        &self.model_list
    }
}
