pub mod adapter;

#[cfg(feature = "tpl-gotpl")]
pub mod golang;
#[cfg(feature = "tpl-gtmpl")]
pub mod gtmpl;
#[cfg(feature = "tpl-minijinja")]
pub mod jinja;

pub use adapter::*;

#[cfg(not(any(
    feature = "tpl-gotpl",
    feature = "tpl-minijinja",
    feature = "tpl-gtmpl"
)))]
compile_error!(
    "No template engine feature enabled. Please enable either 'tpl-gotpl' ,'tpl-minijinja' or 'tpl-gtmpl' feature."
);

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

type RenderFunc = fn(&str, &TemplateData) -> Result<String, Box<dyn std::error::Error>>;

// 工具属性定义
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ToolProperty {
    #[serde(rename = "type")]
    pub property_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", rename = "enum")]
    pub enum_values: Option<Vec<String>>,
}

impl ToolProperty {
    pub fn to_type_script_type(&self) -> String {
        match self.property_type.as_str() {
            "string" => {
                if let Some(enums) = &self.enum_values {
                    if enums.is_empty() {
                        "string".to_string()
                    } else {
                        format!(
                            "{}",
                            enums
                                .iter()
                                .map(|s| format!("\"{}\"", s))
                                .collect::<Vec<_>>()
                                .join(" | ")
                        )
                    }
                } else {
                    "string".to_string()
                }
            }
            "number" => "number".to_string(),
            "integer" => "number".to_string(),
            "boolean" => "boolean".to_string(),
            "array" => "any[]".to_string(),
            "object" => "Record<string, any>".to_string(),
            _ => "any".to_string(),
        }
    }
}

// 工具函数参数定义
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct FunctionParameters {
    #[serde(rename = "type")]
    pub param_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub properties: Option<HashMap<String, ToolProperty>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub required: Option<Vec<String>>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Function {
    #[serde(rename = "name")]
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parameters: Option<FunctionParameters>,
    #[serde(skip_serializing_if = "Option::is_none", rename = "Arguments")]
    pub arguments: Option<Value>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ToolCall {
    #[serde(rename = "Function")]
    pub function: Function,
    #[serde(skip_serializing_if = "Option::is_none", rename = "id")]
    pub id: Option<String>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Tool {
    #[serde(rename = "Type")]
    pub tool_type: String,
    #[serde(rename = "Function")]
    pub function: Function,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Message {
    #[serde(rename = "Role")]
    pub role: String,
    #[serde(skip_serializing_if = "Option::is_none", rename = "Content")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", rename = "ToolCalls")]
    pub tool_calls: Option<Vec<ToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none", rename = "Name")]
    pub name: Option<String>,
}

#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct TemplateData {
    // 基础字段（来自TemplateData）
    #[serde(skip_serializing_if = "Option::is_none", rename = "System")]
    pub system: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", rename = "Tools")]
    pub tools: Option<Vec<Tool>>,
    #[serde(skip_serializing_if = "Option::is_none", rename = "Messages")]
    pub messages: Option<Vec<Message>>,
    #[serde(skip_serializing_if = "Option::is_none", rename = "Prompt")]
    pub prompt: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", rename = "Response")]
    pub response: Option<String>,

    // 扩展字段（来自TemplateValues）
    #[serde(skip_serializing_if = "Option::is_none", rename = "Suffix")]
    pub suffix: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", rename = "Think")]
    pub think: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none", rename = "ThinkLevel")]
    pub think_level: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", rename = "IsThinkSet")]
    pub is_think_set: Option<bool>,

    // 内部使用字段
    #[serde(skip)]
    pub force_legacy: bool,
}

impl TemplateData {
    pub fn new() -> Self {
        Self::default()
    }

    // Builder模式方法
    pub fn with_system(mut self, system: Option<String>) -> Self {
        self.system = system;
        self
    }

    pub fn with_tools(mut self, tools: Option<Vec<Tool>>) -> Self {
        self.tools = tools;
        self
    }

    pub fn with_messages(mut self, messages: Option<Vec<Message>>) -> Self {
        self.messages = messages;
        self
    }

    pub fn with_prompt(mut self, prompt: Option<String>) -> Self {
        self.prompt = prompt;
        self
    }

    pub fn with_response(mut self, response: Option<String>) -> Self {
        self.response = response;
        self
    }

    pub fn with_suffix(mut self, suffix: Option<String>) -> Self {
        self.suffix = suffix;
        self
    }

    pub fn with_think(mut self, think: Option<bool>) -> Self {
        self.think = think;
        self
    }

    pub fn with_think_level(mut self, think_level: Option<String>) -> Self {
        self.think_level = think_level;
        self
    }

    pub fn with_is_think_set(mut self, is_think_set: Option<bool>) -> Self {
        self.is_think_set = is_think_set;
        self
    }

    pub fn with_force_legacy(mut self, force_legacy: bool) -> Self {
        self.force_legacy = force_legacy;
        self
    }
}

// 模板参数结构体
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct TemplateParameters {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<Vec<String>>,
}

// 命名模板结构体
#[derive(Debug, Clone)]
pub struct NamedTemplate {
    pub name: String,
    pub template: String,
    pub bytes: Vec<u8>,
    pub parameters: Option<TemplateParameters>,
}

// 消息合并功能
pub fn collate_messages(messages: Vec<Message>) -> (Option<String>, Vec<Message>) {
    let mut system_parts = Vec::new();
    let mut collated: Vec<Message> = Vec::new();

    for msg in messages {
        if msg.role == "system" {
            if let Some(content) = msg.content {
                system_parts.push(content);
            }
            continue;
        }

        if let Some(last) = collated.last_mut() {
            if last.role == msg.role && msg.role != "tool" {
                if let (Some(last_content), Some(msg_content)) = (&mut last.content, &msg.content) {
                    *last_content = format!("{}\n\n{}", last_content, msg_content);
                    continue;
                }
            }
        }

        collated.push(msg);
    }

    let system = if system_parts.is_empty() {
        None
    } else {
        Some(system_parts.join("\n\n"))
    };

    (system, collated)
}
