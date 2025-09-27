use super::EngineBackend;
use crate::{discover::Model, engine::EngineConfig};
use anyhow::Result;

pub struct TransformersEngine {
    model_info: Model,
    args: EngineConfig,
}

impl EngineBackend for TransformersEngine {
    fn new(args: &EngineConfig, model_info: &Model) -> Result<Self> {
        let mut engine = TransformersEngine {
            model_info: model_info.clone(),
            args: args.clone(),
        };
        Ok(engine)
    }
    fn infer(
        &self,
        prompt: &str,
        option: Option<&EngineConfig>,
        callback: Option<Box<dyn FnMut(String) + Send>>,
    ) -> Result<String> {
        let args = option.unwrap_or(&self.args);
    }

    fn get_model_info(&self) -> Model {
        self.model_info.clone()
    }
}
