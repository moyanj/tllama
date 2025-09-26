use super::EngineBackend;
use crate::{discover::Model, engine::EngineConfig};
use anyhow::Result;
use pyo3::prelude::*;

pub struct TransformersEngine {
    model_info: Model,
    args: EngineConfig,
    model: Option<Py<PyAny>>,
    tokenizer: Option<Py<PyAny>>,
}

impl TransformersEngine {
    fn init_model(&mut self) -> Result<()> {
        self.model = Some(Python::attach(|py| -> PyResult<Py<PyAny>> {
            let auto_model = py.import("transformers")?.getattr("AutoModelForCausalLM")?;
            let from_pretrained = auto_model.getattr("from_pretrained")?;
            from_pretrained
                .call1((self.model_info.model_path.to_str(),))
                .map(|obj| obj.unbind())
        })?);
        self.tokenizer = Some(Python::attach(|py| -> PyResult<Py<PyAny>> {
            let auto_model = py.import("transformers")?.getattr("AutoTokenizer")?;
            let from_pretrained = auto_model.getattr("from_pretrained")?;
            from_pretrained
                .call1((self.model_info.model_path.to_str(),))
                .map(|obj| obj.unbind())
        })?);
        Ok(())
    }
}

impl EngineBackend for TransformersEngine {
    fn new(args: &EngineConfig, model_info: &Model) -> Result<Self> {
        let mut engine = TransformersEngine {
            model_info: model_info.clone(),
            args: args.clone(),
            model: None,
            tokenizer: None,
        };
        engine.init_model()?;
        Ok(engine)
    }
    fn infer(
        &self,
        prompt: &str,
        option: Option<&EngineConfig>,
        callback: Option<Box<dyn FnMut(String) + Send>>,
    ) -> Result<String> {
        todo!()
    }

    fn get_model_info(&self) -> Model {
        self.model_info.clone()
    }
}
