use crate::{
    discover::{Model, ModelType},
    engine::{EngineBackend, EngineConfig},
};
use anyhow::Result;

#[cfg(feature = "engine-hf")]
use super::hf::TransformersEngine;
#[cfg(feature = "engine-llama-cpp")]
use super::llama_cpp::LlamaEngine;
pub struct InferenceEngine {
    engine: Box<dyn EngineBackend + Send>,
}

impl InferenceEngine {
    pub fn new(args: &EngineConfig, model: &Model) -> Result<Self, Box<dyn std::error::Error>> {
        Ok(InferenceEngine {
            engine: match model.model_type {
                #[cfg(feature = "engine-llama-cpp")]
                ModelType::Gguf => Box::new(LlamaEngine::new(args, model)?),
                #[cfg(feature = "engine-hf")]
                ModelType::Safetensors => Box::new(TransformersEngine::new(args, model)?),
                _ => panic!("Unsupported model type"),
            },
        })
    }

    pub fn infer(
        &self,
        prompt: &str,
        option: Option<&EngineConfig>,
        callback: Option<Box<dyn FnMut(String) + Send>>,
    ) -> Result<String> {
        self.engine.infer(prompt, option, callback)
    }

    pub fn get_model_info(&self) -> Model {
        self.engine.get_model_info()
    }
}
