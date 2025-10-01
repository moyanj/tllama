#[cfg(feature = "api")]
pub mod api;
#[cfg(feature = "chat")]
pub mod chat;
pub mod cli;
pub mod discover;
pub mod engine;
pub mod template;

//#[cfg(feature = "engine-hf")]
//compile_error!("The `engine-hf` feature is not supported yet.");
