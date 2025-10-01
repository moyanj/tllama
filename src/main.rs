use clap::Parser;
use std::io::Write;
use tllama::cli;
use tllama::def_callback;
use tllama::discover;
use tllama::discover::Model;
use tllama::engine::{EngineConfig, InferenceEngine};
use tracing_subscriber::EnvFilter;

async fn serve(args: &cli::ServeArgs) -> Result<(), Box<dyn std::error::Error>> {
    tllama::api::start_api_server(args.host.clone(), args.port).await?;
    Ok(())
}

fn infer(args: &cli::InferArgs) -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(feature = "engine-llama-cpp")]
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

    let engine = InferenceEngine::new(
        &EngineConfig {
            n_ctx: args.n_ctx.unwrap_or(4096),
            n_len: args.n_len,
            temperature: args.temperature.unwrap_or(0.8),
            top_k: args.top_k.unwrap_or(40),
            top_p: args.top_p.unwrap_or(0.9),
            repeat_penalty: args.repeat_penalty.unwrap_or(1.1),
        },
        &model_path,
    )?;

    engine.infer(
        &prompt,
        None,
        def_callback!(|token| {
            print!("{}", token);
            std::io::stdout().flush().unwrap();
        }),
    )?;
    println!();
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
            let model_type = match model.format {
                discover::ModelType::Gguf => "GGUF",
                discover::ModelType::Transformers => "Transformers",
            };

            // 智能单位显示
            let (size_str, unit) = if model.size < 1024 * 1024 * 1024 {
                (
                    format!("{:.2}", model.size as f64 / (1024.0 * 1024.0)),
                    "MB",
                )
            } else {
                (
                    format!("{:.2}", model.size as f64 / (1024.0 * 1024.0 * 1024.0)),
                    "GB",
                )
            };

            println!("Name: {}", model.name);
            println!("  Path: {}", model.path.display());
            println!("  Type: {}", model_type);
            println!("  Size: {} {}", size_str, unit);
            println!();
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
            let discoverer_result = discover::MODEL_DISCOVERER.lock();
            let mut discoverer = match discoverer_result {
                Ok(discoverer) => discoverer,
                Err(err) => {
                    eprintln!("Error: {}", err);
                    return Ok(());
                }
            };
            discoverer.scan_all_paths(args.all);
            discoverer.discover();
        }
        cli::Commands::List => {
            list_models()?;
        }
        #[cfg(feature = "chat")]
        cli::Commands::Chat(args) => {
            tllama::chat::chat_session(args)?;
        }
        cli::Commands::Serve(args) => {
            serve(&args).await?;
        }
    }
    Ok(())
}
