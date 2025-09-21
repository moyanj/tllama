use std::{collections::HashMap, error::Error, sync::Arc};
use tokio::sync::Mutex;

use crate::{
    discover::MODEL_DISCOVERER,
    engine::{EngineConfig, InferenceEngine},
};

pub struct ModelPool {
    // 使用 Mutex 包裹 HashMap 以实现内部可变性
    // HashMap 的值类型改为 Arc<dyn InferenceEngine> 来存储特征对象并允许共享
    models: Mutex<HashMap<String, Arc<dyn InferenceEngine>>>,
}

impl ModelPool {
    pub fn new() -> Self {
        ModelPool {
            models: Mutex::new(HashMap::new()),
        }
    }

    pub async fn get_model(
        &self,
        model_name: &str,
    ) -> Result<Arc<dyn InferenceEngine>, Box<dyn Error>> {
        // 1. 尝试从池中获取模型，如果存在则直接返回
        {
            let models_guard = self.models.lock().await; // 异步锁
            if let Some(engine) = models_guard.get(model_name) {
                println!("[ModelPool] Model '{}' found in pool.", model_name);
                return Ok(Arc::clone(engine));
            }
        } // `models_guard` 在这里超出作用域，释放了锁。

        println!(
            "[ModelPool] Model '{}' not found in pool. Discovering and loading...",
            model_name
        );

        // 2. 如果模型不在池中，则需要发现并加载它
        // 在发现和加载模型期间，我们不持有 `self.models` 的锁，以避免阻塞其他请求。
        let model_path_string = {
            let discoverer_guard = MODEL_DISCOVERER.lock().unwrap(); // 阻塞式锁，但很快就会释放
            discoverer_guard
                .find_model(model_name)
                .map_err(|e| -> Box<dyn Error> {
                    Box::new(std::io::Error::new(
                        std::io::ErrorKind::Other,
                        format!("Model '{}' not found: {}", model_name, e),
                    )) as Box<dyn Error>
                })?
        };

        // 定义用于加载模型的默认 EngineConfig。
        // 这些值与 `main.rs` 中 `infer` 命令使用的默认值相匹配。
        let engine_config = EngineConfig {
            n_ctx: 2048,
            n_len: 2048,
            temperature: 0.8,
            top_k: 40,
            top_p: 0.9,
            repeat_penalty: 1.1,
        };

        // 加载 LlamaEngine。这是一个可能耗时的操作。
        let engine = crate::engine::llama_cpp::LlamaEngine::new(&engine_config, &model_path_string)
            .map_err(|e| -> Box<dyn Error> {
                Box::new(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    format!("Failed to load model '{}': {}", model_name, e),
                )) as Box<dyn Error>
            })?;

        // 将加载的引擎封装在 `Arc<dyn InferenceEngine>>` 中
        let new_engine_arc: Arc<dyn InferenceEngine> = Arc::new(engine);

        // 3. 将新加载的模型添加到池中
        let mut models_guard = self.models.lock().await; // 重新获取锁以修改 HashMap
        models_guard.insert(model_name.to_string(), Arc::clone(&new_engine_arc));

        println!(
            "[ModelPool] Model '{}' loaded and added to pool.",
            model_name
        );
        Ok(new_engine_arc)
    }
}
