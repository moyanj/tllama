mod cli;
mod engine;

use clap::Parser;
use engine::llama_cpp::LlamaEngine;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = cli::Cli::parse();
    let mut engine = LlamaEngine::new(&args)?;
    let result = engine.infer(&args.prompt)?;
    println!("{}", result);
    Ok(())
}
