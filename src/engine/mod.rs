use crate::discover::Model;

pub trait InferenceEngine: Send + Sync {
    fn infer(&mut self, prompt: &str) -> Result<String, Box<dyn std::error::Error>>;
    fn infer_stream(
        &mut self,
        prompt: &str,
        callback: &mut dyn FnMut(&str) -> Result<(), Box<dyn std::error::Error>>,
    ) -> Result<(), Box<dyn std::error::Error>>;
    fn get_model_info(&self) -> Model;
    fn set_config(&mut self, _config: &EngineConfig) {}
}

#[derive(Clone, Debug)]
pub struct EngineConfig {
    pub n_ctx: i32,
    pub n_len: u32,
    pub temperature: f32,
    pub top_k: i32,
    pub top_p: f32,
    pub repeat_penalty: f32,
}

#[cfg(feature = "engine-llama-cpp")]
pub mod llama_cpp;

#[cfg(not(any(feature = "engine-llama-cpp", feature = "engine-hf")))]
compile_error!(
    "No template engine feature enabled. Please enable either 'engine-llama-cpp' or 'engine-hf'."
);
