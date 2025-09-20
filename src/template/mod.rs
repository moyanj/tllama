#[cfg(feature = "tpl-gotpl")]
pub mod golang;
#[cfg(feature = "tpl-minijinja")]
pub mod jinja;

#[cfg(feature = "tpl-gotpl")]
pub use golang::*;
#[cfg(feature = "tpl-minijinja")]
pub use jinja::*;

use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Function {
    #[serde(rename = "Name")]
    pub name: String,
    #[serde(rename = "Arguments")]
    pub arguments: Value,
}
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ToolCall {
    #[serde(rename = "Function")]
    pub function: Function,
}
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Tool {
    #[serde(rename = "Function")]
    pub function: Value,
}
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Message {
    #[serde(rename = "Role")]
    pub role: String,
    #[serde(skip_serializing_if = "Option::is_none", rename = "Content")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", rename = "ToolCalls")]
    pub tool_calls: Option<Vec<ToolCall>>,
}
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct PromptData {
    #[serde(skip_serializing_if = "Option::is_none", rename = "System")]
    pub system: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", rename = "Tools")]
    pub tools: Option<Vec<Tool>>,
    #[serde(skip_serializing_if = "Option::is_none", rename = "Messages")]
    pub messages: Option<Vec<Message>>,
    #[serde(skip_serializing_if = "Option::is_none", rename = "Prompt")]
    pub prompt: Option<String>,
    #[serde(rename = "Response")]
    pub response: Option<String>,
}
