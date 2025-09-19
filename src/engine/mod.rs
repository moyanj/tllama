pub trait InferenceEngine {
    fn infer(&mut self, prompt: &str) -> Result<String, Box<dyn std::error::Error>>;
}

pub mod llama_cpp;