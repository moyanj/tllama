use clap::{Parser, Subcommand};

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand, Debug)]
pub enum Commands {
    Infer(InferArgs),
    Discover(DiscoverArgs),
    List,
    #[cfg(feature = "chat")]
    Chat(ChatArgs),
}

#[derive(Parser, Debug)]
pub struct InferArgs {
    pub model: String,
    pub prompt: String,
    #[arg(short, long)]
    /// Number of tokens to generate
    pub n_len: Option<u32>,
    #[arg(short, long)]
    /// Sampling temperature
    /// Typical values are between 0.1 and 1.0
    pub temperature: Option<f32>,
    #[arg(short = 'k', long)]
    /// Top-k sampling
    /// Typical values are between 1 and 100
    pub top_k: Option<i32>,
    #[arg(short = 'p', long)]
    /// Top-p (nucleus) sampling
    /// Typical values are between 0.5 and 1.0
    pub top_p: Option<f32>,
    #[arg(short = 'r', long)]
    /// Repeat penalty
    /// Typical values are between 1.0 and 2.0
    pub repeat_penalty: Option<f32>,
    #[arg(short = 'c', long)]
    /// Context size
    /// Typical values are 512, 1024, 2048, etc.
    pub n_ctx: Option<i32>,
}

#[derive(Parser, Debug)]
pub struct DiscoverArgs {
    #[arg(short, long)]
    /// Scan all paths
    pub all: bool,
}

#[cfg(feature = "chat")]
#[derive(Parser, Debug)]
pub struct ChatArgs {
    pub model: String,
}
