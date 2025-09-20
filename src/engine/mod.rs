pub trait InferenceEngine {
    fn infer(&mut self, prompt: &str) -> Result<String, Box<dyn std::error::Error>>;
    fn infer_stream(
        &mut self,
        prompt: &str,
        callback: impl FnMut(&str) -> Result<(), Box<dyn std::error::Error>>,
    ) -> Result<(), Box<dyn std::error::Error>>;
}

pub struct EngineConfig {
    pub n_ctx: i32,
    pub n_len: u32,
    pub temperature: f32,
    pub top_k: i32,
    pub top_p: f32,
    pub repeat_penalty: f32,
}

pub mod llama_cpp;
