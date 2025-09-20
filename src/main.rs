mod cli;

use clap::Parser;
use rllama::engine::{EngineConfig, InferenceEngine, llama_cpp::LlamaEngine};
use rllama::template::*;
use std::io::Write;
/*
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = cli::Cli::parse();

    let example_data_1 = PromptData {
        system: Some("You are a helpful AI assistant.".to_string()),
        tools: None,
        messages: Some(vec![Message {
            role: "user".to_string(),
            content: Some(args.prompt.to_string()),
            tool_calls: None,
        }]),
        prompt: None,
        response: None,
    };
    let prompt = render_chatml_template(&example_data_1)?;
    println!("Rendered Prompt:\n{}", prompt);
    //exit(1);
    let mut engine = LlamaEngine::new(
        &EngineConfig {
            n_ctx: args.n_ctx.unwrap_or(2048),
            n_len: args.n_len.unwrap_or(2048),
            temperature: args.temperature.unwrap_or(0.8),
            top_k: args.top_k.unwrap_or(40),
            top_p: args.top_p.unwrap_or(0.9), // 修改默认值为0.9
            repeat_penalty: args.repeat_penalty.unwrap_or(1.1),
        },
        &args.model_path,
    )?;
    if args.stream {
        // 流式模式 - 逐词输出
        engine.infer_stream(&prompt, |token| {
            print!("{}", token);
            std::io::stdout().flush()?; // 立即刷新输出
            Ok(())
        })?;

        println!(); // 输出完成后换行
    } else {
        // 批量模式 - 一次性输出
        let result = engine.infer(&prompt)?;
        println!("{}", result);
    }
    Ok(())
}*/

fn main() {
    let mut discoverer = rllama::discover::MODEL_DISCOVERER.lock().unwrap();
    discoverer.scan_all_paths();
    discoverer.discover();
    // 打印发现的模型
    let models = discoverer.get_model_list();
    println!("发现 {} 个模型:", models.len());
    for model in models {
        println!("- 名称: {}, 路径: {:?}", model.model_name, model.model_path);
    }
}
