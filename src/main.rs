use clap::Parser;
use rllama::cli;
use rllama::def_callback;
use rllama::discover;
use rllama::discover::Model;
use rllama::engine::{EngineConfig, InferenceEngine, llama_cpp::LlamaEngine};
use std::io::Write;
use tracing_subscriber::EnvFilter;

async fn serve(args: &cli::ServeArgs) -> Result<(), Box<dyn std::error::Error>> {
    // The server now starts with an empty model pool.
    // Models are loaded dynamically via the API.
    rllama::api::start_api_server(args.host.clone(), args.port).await?;
    Ok(())
}

fn infer(args: &cli::InferArgs) -> Result<(), Box<dyn std::error::Error>> {
    llama_cpp_2::send_logs_to_tracing(llama_cpp_2::LogOptions::default().with_logs_enabled(false));
    let prompt = args.prompt.clone();

    let model_path;
    if args.model.starts_with(".") || args.model.starts_with("/") {
        model_path = Model::from_path(&args.model);
    } else {
        model_path = discover::MODEL_DISCOVERER
            .lock()
            .unwrap()
            .find_model(&args.model)?;
    }
    //exit(1);
    let engine = LlamaEngine::new(
        &EngineConfig {
            n_ctx: args.n_ctx.unwrap_or(4096),
            n_len: args.n_len,
            temperature: args.temperature.unwrap_or(0.8),
            top_k: args.top_k.unwrap_or(40),
            top_p: args.top_p.unwrap_or(0.9), // 修改默认值为0.9
            repeat_penalty: args.repeat_penalty.unwrap_or(1.1),
        },
        &model_path,
    )?;
    // 流式模式 - 逐词输出
    engine.infer(
        &prompt,
        None,
        def_callback!(|token| {
            print!("{}", token);
            std::io::stdout().flush().unwrap();
        }),
    )?;
    Ok(())
}

fn list_models() -> Result<(), Box<dyn std::error::Error>> {
    let discoverer = discover::MODEL_DISCOVERER.lock().unwrap();
    let models = discoverer.get_model_list();
    if models.is_empty() {
        println!("No models found.");
    } else {
        println!("Discovered Models:");
        for model in models {
            let model_type = match model.model_type {
                discover::ModelType::Gguf => "GGUF",
                discover::ModelType::Safetensors => "Safetensors",
            };

            // 智能单位显示
            let (size_str, unit) = if model.model_size < 1024 * 1024 * 1024 {
                (
                    format!("{:.2}", model.model_size as f64 / (1024.0 * 1024.0)),
                    "MB",
                )
            } else {
                (
                    format!(
                        "{:.2}",
                        model.model_size as f64 / (1024.0 * 1024.0 * 1024.0)
                    ),
                    "GB",
                )
            };

            println!("Name: {}", model.model_name);
            println!("  Path: {}", model.model_path.display());
            println!("  Type: {}", model_type);
            println!("  Size: {} {}", size_str, unit);
            println!(); // 添加空行以分隔不同模型
        }
    }
    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();

    let args = cli::Cli::parse();
    match args.command {
        cli::Commands::Infer(args) => {
            infer(&args)?;
        }
        cli::Commands::Discover(args) => {
            discover::MODEL_DISCOVERER
                .lock()
                .unwrap()
                .scan_all_paths(args.all);
            discover::MODEL_DISCOVERER.lock().unwrap().discover();
        }
        cli::Commands::List => {
            list_models()?;
        }
        #[cfg(feature = "chat")]
        cli::Commands::Chat(args) => {
            rllama::chat::chat_session(args)?;
        }
        cli::Commands::Serve(args) => {
            serve(&args).await?;
        }
    }
    Ok(())
}
