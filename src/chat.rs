use std::io::Write;

use crate::{
    engine::{EngineConfig, InferenceEngine},
    template::*,
};

struct ChatSession {
    engine: Box<dyn InferenceEngine>,
    data: Vec<Message>,
    system_prompt: String,
}

impl ChatSession {
    fn new(engine: Box<dyn InferenceEngine>) -> Self {
        Self {
            engine,
            data: vec![],
            system_prompt: "You are a helpful AI assistant.".to_string(),
        }
    }

    fn start(&mut self) {
        loop {
            print!(">>>");
            std::io::stdout().flush().unwrap();
            let mut input = String::new();
            std::io::stdin()
                .read_line(&mut input)
                .expect("Failed to read line");
            let input = input.trim();

            if input == ".exit" || input == ".quit" || input == ".q" || input == ".bye" {
                break;
            }

            self.chat(input).unwrap();
            println!("");
        }
    }

    fn chat(&mut self, user_input: &str) -> Result<(), Box<dyn std::error::Error>> {
        self.data.push(Message {
            role: "user".to_string(),
            content: Some(user_input.to_string()),
            tool_calls: None,
        });

        let prompt_data = PromptData {
            system: Some(self.system_prompt.clone()),
            tools: None,
            messages: Some(self.data.clone()),
            prompt: None,
            response: None,
        };

        let prompt = render_chatml_template(&prompt_data)?;
        let mut out = vec![];
        self.engine.infer_stream(&prompt, &mut |token| {
            print!("{}", token);
            std::io::stdout().flush()?;
            out.push(token.to_string());
            Ok(())
        })?;
        self.data.push(Message {
            role: "assistant".to_string(),
            content: Some(out.join("")),
            tool_calls: None,
        });
        Ok(())
    }
}

pub fn chat_session(args: crate::cli::ChatArgs) -> Result<(), Box<dyn std::error::Error>> {
    let model_path;
    if args.model.starts_with(".") || args.model.starts_with("/") {
        model_path = args.model.clone();
    } else {
        model_path = crate::discover::MODEL_DISCOVERER
            .lock()
            .unwrap()
            .find_model(&args.model)?;
    }

    let engine = Box::new(crate::engine::llama_cpp::LlamaEngine::new(
        &EngineConfig {
            n_ctx: 2048,
            n_len: 2048,
            temperature: 0.8,
            top_k: 40,
            top_p: 0.9, // 修改默认值为0.9
            repeat_penalty: 1.1,
        },
        &model_path,
    )?);

    let mut session = ChatSession::new(engine);
    session.start();
    Ok(())
}
