use crate::discover::Model;
use anyhow::Result;

type EngineCallback = Box<dyn FnMut(String) -> bool + Send>;

pub trait EngineBackend: Send + Sync {
    fn new(args: &EngineConfig, model: &Model) -> Result<Self>
    where
        Self: Sized;
    fn infer(
        &self,
        prompt: &str,
        option: Option<&EngineConfig>,
        callback: Option<EngineCallback>,
    ) -> Result<String>;
    fn get_model_info(&self) -> Model;
}

#[macro_export]
macro_rules! def_callback {
    (|$arg:ident| $body:block) => {
        Some(Box::new(move |$arg: String| $body) as Box<dyn FnMut(String) -> bool + Send + 'static>)
    };
}

#[derive(Clone, Debug, Serialize)]
pub struct EngineConfig {
    pub n_ctx: i32,
    pub n_len: Option<u32>,
    pub temperature: f32,
    pub top_k: i32,
    pub top_p: f32,
    pub repeat_penalty: f32,
}

impl Default for EngineConfig {
    fn default() -> Self {
        EngineConfig {
            n_ctx: 4096,
            n_len: None,
            temperature: 0.7,
            top_k: 40,
            top_p: 0.9,
            repeat_penalty: 1.1,
        }
    }
}

pub mod adapter;

pub use adapter::InferenceEngine;
use serde::Serialize;

#[cfg(feature = "engine-llama-cpp")]
pub mod llama_cpp;

#[cfg(feature = "engine-hf")]
pub mod hf;

#[cfg(not(any(feature = "engine-llama-cpp", feature = "engine-hf")))]
compile_error!(
    "No template engine feature enabled. Please enable either 'engine-llama-cpp' or 'engine-hf'."
);
