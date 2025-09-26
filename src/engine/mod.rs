use crate::discover::Model;
use anyhow::Result;

pub trait EngineBackend: Send + Sync {
    fn infer(
        &self,
        prompt: &str,
        option: Option<&EngineConfig>,
        callback: Option<Box<dyn FnMut(String) + Send>>,
    ) -> Result<String>;
    fn get_model_info(&self) -> Model;
}

#[macro_export]
macro_rules! def_callback {
    (|$arg:ident| $body:block) => {
        Some(Box::new(move |$arg: String| $body) as Box<dyn FnMut(String) + Send + 'static>)
    };
}

#[derive(Clone, Debug)]
pub struct EngineConfig {
    pub n_ctx: i32,
    pub n_len: Option<u32>,
    pub temperature: f32,
    pub top_k: i32,
    pub top_p: f32,
    pub repeat_penalty: f32,
}

pub mod adapter;

pub use adapter::InferenceEngine;

#[cfg(feature = "engine-llama-cpp")]
pub mod llama_cpp;

#[cfg(not(any(feature = "engine-llama-cpp", feature = "engine-hf")))]
compile_error!(
    "No template engine feature enabled. Please enable either 'engine-llama-cpp' or 'engine-hf'."
);
