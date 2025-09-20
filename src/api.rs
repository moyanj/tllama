use axum::{
    Json, Router,
    extract::State,
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
};
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};
use tokio::net::TcpListener;

use crate::{
    engine::{EngineConfig, InferenceEngine},
    template::{Message, PromptData, render_chatml_template},
};

#[derive(Clone)]
struct AppState {
    engine: Arc<Mutex<Box<dyn InferenceEngine + Send>>>,
    system_prompt: String,
}

// OpenAI 兼容的请求结构
#[derive(Debug, Deserialize)]
struct OpenAIRequest {
    model: String,
    messages: Vec<OpenAIMessage>,
    temperature: Option<f32>,
    max_tokens: Option<u32>,
    stream: Option<bool>,
}

#[derive(Debug, Deserialize, Serialize)]
struct OpenAIMessage {
    role: String,
    content: String,
}

// OpenAI 兼容的响应结构
#[derive(Debug, Serialize)]
struct OpenAIResponse {
    id: String,
    object: String,
    created: u64,
    model: String,
    choices: Vec<OpenAIChoice>,
    usage: OpenAIUsage,
}

#[derive(Debug, Serialize)]
struct OpenAIChoice {
    index: u32,
    message: OpenAIMessage,
    finish_reason: String,
}

#[derive(Debug, Serialize)]
struct OpenAIUsage {
    prompt_tokens: u32,
    completion_tokens: u32,
    total_tokens: u32,
}

// Rllama 控制端点的请求结构
#[derive(Debug, Deserialize)]
struct RllamaRequest {
    prompt: String,
    temperature: Option<f32>,
    max_tokens: Option<u32>,
    top_k: Option<i32>,
    top_p: Option<f32>,
    repeat_penalty: Option<f32>,
}

// Rllama 控制端点的响应结构
#[derive(Debug, Serialize)]
struct RllamaResponse {
    text: String,
}

// 健康检查端点
async fn health() -> impl IntoResponse {
    (StatusCode::OK, "OK")
}

// OpenAI 兼容的聊天端点
async fn openai_chat(
    State(state): State<AppState>,
    Json(request): Json<OpenAIRequest>,
) -> impl IntoResponse {
    let mut engine = state.engine.lock().unwrap();
    let system_prompt = state.system_prompt.clone();

    // 转换消息格式
    let messages: Vec<Message> = request
        .messages
        .into_iter()
        .map(|msg| Message {
            role: msg.role,
            content: Some(msg.content),
            tool_calls: None,
        })
        .collect();

    let prompt_data = PromptData {
        system: Some(system_prompt),
        tools: None,
        messages: Some(messages),
        prompt: None,
        response: None,
    };

    let prompt = match render_chatml_template(&prompt_data) {
        Ok(p) => p,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Template error: {}", e),
            )
                .into_response();
        }
    };

    let n_len = request.max_tokens.unwrap_or(2048);

    // 设置临时引擎配置
    let temp_config = EngineConfig {
        n_ctx: 2048,
        n_len,
        temperature: request.temperature.unwrap_or(0.8),
        top_k: 40,
        top_p: 0.9,
        repeat_penalty: 1.1,
    };

    // 执行推理
    let result = match engine.infer(&prompt) {
        Ok(output) => output,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Inference error: {}", e),
            )
                .into_response();
        }
    };

    // 构建响应
    let response = OpenAIResponse {
        id: "chatcmpl-123".to_string(),
        object: "chat.completion".to_string(),
        created: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs(),
        model: request.model,
        choices: vec![OpenAIChoice {
            index: 0,
            message: OpenAIMessage {
                role: "assistant".to_string(),
                content: result,
            },
            finish_reason: "stop".to_string(),
        }],
        usage: OpenAIUsage {
            prompt_tokens: 0,     // 需要实际计算
            completion_tokens: 0, // 需要实际计算
            total_tokens: 0,      // 需要实际计算
        },
    };

    (StatusCode::OK, Json(response)).into_response()
}

// Rllama 控制端点
async fn rllama_complete(
    State(state): State<AppState>,
    Json(request): Json<RllamaRequest>,
) -> impl IntoResponse {
    let mut engine = state.engine.lock().unwrap();

    let n_len = request.max_tokens.unwrap_or(2048);

    // 设置临时引擎配置
    let temp_config = EngineConfig {
        n_ctx: 2048,
        n_len,
        temperature: request.temperature.unwrap_or(0.8),
        top_k: request.top_k.unwrap_or(40),
        top_p: request.top_p.unwrap_or(0.9),
        repeat_penalty: request.repeat_penalty.unwrap_or(1.1),
    };

    // 执行推理
    let result = match engine.infer(&request.prompt) {
        Ok(output) => output,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Inference error: {}", e),
            )
                .into_response();
        }
    };

    let response = RllamaResponse { text: result };
    (StatusCode::OK, Json(response)).into_response()
}

pub async fn start_api_server(
    engine: Box<dyn InferenceEngine + Send>,
    host: String,
    port: u16,
    system_prompt: Option<String>,
) -> Result<(), Box<dyn std::error::Error>> {
    let state = AppState {
        engine: Arc::new(Mutex::new(engine)),
        system_prompt: system_prompt.unwrap_or_else(|| {
            "You are a helpful, respectful and honest AI assistant.".to_string()
        }),
    };

    let app = Router::new()
        .route("/health", get(health))
        .route("/v1/chat/completions", post(openai_chat))
        .route("/rllama/complete", post(rllama_complete))
        .with_state(state);

    let addr = format!("{}:{}", host, port);
    let listener = TcpListener::bind(&addr).await?;
    println!("API server listening on {}", addr);

    axum::serve(listener, app).await?;
    Ok(())
}
