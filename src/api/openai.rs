// src/api/openai_compatible.rs
use super::server::AppState;
use crate::template::Message;
use crate::{discover::MODEL_DISCOVERER, engine::EngineConfig};
use actix_web::web::Bytes;
use actix_web::{HttpResponse, Result as ActixResult, web};
use serde::Serialize;
use serde_json::{Value, json};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio_stream::StreamExt;
use uuid::Uuid;

// OpenAI 兼容的请求结构体
#[derive(serde::Deserialize, Debug)]
pub struct CompletionRequest {
    pub model: String,
    pub prompt: String,
    #[serde(default)]
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    #[serde(default)]
    pub n: Option<u32>,
    pub stream: Option<bool>,
    pub stop: Option<Vec<String>>,
    pub presence_penalty: Option<f32>,
    pub frequency_penalty: Option<f32>,
    pub logprobs: Option<u32>,
    pub echo: Option<bool>,
    pub suffix: Option<String>,
    pub best_of: Option<u32>,
}

#[derive(serde::Deserialize, Debug)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(default)]
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    #[serde(default)]
    pub n: Option<u32>,
    pub stream: Option<bool>,
    pub stop: Option<Vec<String>>,
    pub presence_penalty: Option<f32>,
    pub frequency_penalty: Option<f32>,
    pub logit_bias: Option<Value>,
}

#[derive(serde::Deserialize, Debug, Clone, Serialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

// OpenAI 兼容的响应结构体
#[derive(serde::Serialize)]
pub struct ListModelsResponse {
    pub object: String,
    pub data: Vec<ModelData>,
}

#[derive(serde::Serialize)]
pub struct ModelData {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub owned_by: String,
}

#[derive(serde::Serialize)]
pub struct CompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<CompletionChoice>,
    pub usage: Usage,
}

#[derive(serde::Serialize)]
pub struct CompletionChoice {
    pub text: String,
    pub index: u32,
    pub logprobs: Option<Value>,
    pub finish_reason: String,
}

#[derive(serde::Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatCompletionChoice>,
    pub usage: Usage,
}

#[derive(serde::Serialize)]
pub struct ChatCompletionChoice {
    pub index: u32,
    pub message: ChatMessage,
    pub finish_reason: String,
}

#[derive(serde::Serialize)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

// 流式响应结构体
#[derive(serde::Serialize)]
pub struct StreamCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<StreamCompletionChoice>,
}

#[derive(serde::Serialize)]
pub struct StreamCompletionChoice {
    pub text: String,
    pub index: u32,
    pub logprobs: Option<Value>,
    pub finish_reason: Option<String>,
}

#[derive(serde::Serialize)]
pub struct StreamChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<StreamChatCompletionChoice>,
}

#[derive(serde::Serialize)]
pub struct StreamChatCompletionChoice {
    pub index: u32,
    pub delta: ChatMessage,
    pub finish_reason: Option<String>,
}

// 错误响应
#[derive(serde::Serialize)]
pub struct ErrorResponse {
    pub error: ErrorInfo,
}

#[derive(serde::Serialize)]
pub struct ErrorInfo {
    pub message: String,
    #[serde(rename = "type")]
    pub error_type: String,
    pub code: Option<String>,
}

impl From<String> for ErrorInfo {
    fn from(message: String) -> Self {
        ErrorInfo {
            message,
            error_type: "internal_error".to_string(),
            code: None,
        }
    }
}

// OpenAI 兼容的API实现
pub async fn list_models() -> ActixResult<HttpResponse> {
    let models = match MODEL_DISCOVERER.lock() {
        Ok(discoverer) => discoverer.get_model_list().clone(),
        Err(e) => {
            return Ok(HttpResponse::InternalServerError().json(ErrorResponse {
                error: ErrorInfo {
                    message: format!("Failed to acquire model discoverer lock: {}", e),
                    error_type: "internal_error".to_string(),
                    code: Some("lock_error".to_string()),
                },
            }));
        }
    };

    let model_data: Vec<ModelData> = models
        .iter()
        .map(|model| ModelData {
            id: model.name.clone(),
            object: "model".to_string(),
            created: UNIX_EPOCH.elapsed().unwrap().as_secs(),
            owned_by: "tllama".to_string(),
        })
        .collect();

    let response = ListModelsResponse {
        object: "list".to_string(),
        data: model_data,
    };

    Ok(HttpResponse::Ok().json(response))
}

pub async fn create_completion(
    request: web::Json<CompletionRequest>,
    data: web::Data<AppState>,
) -> ActixResult<HttpResponse> {
    let stream_requested = request.stream.unwrap_or(false);
    let model_name = request.model.clone();

    // 转换参数到引擎配置
    let engine_config = EngineConfig {
        n_ctx: 4096, // 默认上下文长度
        n_len: request.max_tokens,
        temperature: request.temperature.unwrap_or(1.0),
        top_k: 40, // OpenAI 使用 top_p，但我们保留 top_k 作为默认
        top_p: request.top_p.unwrap_or(1.0),
        repeat_penalty: 1.0, // 默认不使用重复惩罚
    };

    let engine_arc = match data.model_pool.get_model(&model_name).await {
        Ok(engine) => engine,
        Err(e) => {
            return Ok(HttpResponse::BadRequest().json(ErrorResponse {
                error: ErrorInfo {
                    message: format!("Model not found: {}", e),
                    error_type: "invalid_request_error".to_string(),
                    code: Some("model_not_found".to_string()),
                },
            }));
        }
    };

    if stream_requested {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<StreamCompletionResponse>();
        let prompt = request.prompt.clone();
        let model_name_clone = model_name.clone();
        let engine_arc_clone = Arc::clone(&engine_arc);

        tokio::task::spawn_blocking(move || {
            let tx_tokens = tx.clone();
            let model_name_clone2 = model_name_clone.clone();
            let request_id = Uuid::new_v4().to_string();
            let request_id_clone = request_id.clone();
            let created = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs();

            // 发送初始空响应（如果需要）
            let _ = tx_tokens.send(StreamCompletionResponse {
                id: request_id.clone(),
                object: "text_completion".to_string(),
                created,
                model: model_name_clone.clone(),
                choices: vec![StreamCompletionChoice {
                    text: String::new(),
                    index: 0,
                    logprobs: None,
                    finish_reason: None,
                }],
            });

            let mut accumulated_text = String::new();

            // 执行推理并流式发送响应
            let _ = engine_arc_clone.infer(
                &prompt,
                Some(&engine_config),
                Some(Box::new(move |tok| {
                    accumulated_text.push_str(&tok);
                    let response = StreamCompletionResponse {
                        id: request_id.clone(),
                        object: "text_completion".to_string(),
                        created,
                        model: model_name_clone.clone(),
                        choices: vec![StreamCompletionChoice {
                            text: tok.into(),
                            index: 0,
                            logprobs: None,
                            finish_reason: None,
                        }],
                    };
                    let a = tx_tokens.send(response);
                    if a.is_err() {
                        return false;
                    }
                    true
                })),
            );

            // 发送结束信号
            let _ = tx.send(StreamCompletionResponse {
                id: request_id_clone,
                object: "text_completion".to_string(),
                created,
                model: model_name_clone2,
                choices: vec![StreamCompletionChoice {
                    text: String::new(),
                    index: 0,
                    logprobs: None,
                    finish_reason: Some("stop".to_string()),
                }],
            });
        });

        let stream = tokio_stream::wrappers::UnboundedReceiverStream::new(rx);
        let stream = stream.map(|chunk| {
            let json_str = serde_json::to_string(&chunk).unwrap();
            Ok::<Bytes, actix_web::Error>(Bytes::from(format!("data: {}\n\n", json_str)))
        });

        Ok(HttpResponse::Ok()
            .append_header(("Content-Type", "text/event-stream"))
            .append_header(("Cache-Control", "no-cache"))
            .append_header(("Access-Control-Allow-Origin", "*"))
            .streaming(stream))
    } else {
        // 非流式推理
        match engine_arc.infer(&request.prompt, Some(&engine_config), None) {
            Ok(text) => {
                let response = CompletionResponse {
                    id: Uuid::new_v4().to_string(),
                    object: "text_completion".to_string(),
                    created: SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .unwrap()
                        .as_secs(),
                    model: model_name,
                    choices: vec![CompletionChoice {
                        text,
                        index: 0,
                        logprobs: None,
                        finish_reason: "stop".to_string(),
                    }],
                    usage: Usage {
                        prompt_tokens: 0,     // 需要实际统计
                        completion_tokens: 0, // 需要实际统计
                        total_tokens: 0,
                    },
                };
                Ok(HttpResponse::Ok().json(response))
            }
            Err(e) => Ok(HttpResponse::InternalServerError().json(ErrorResponse {
                error: ErrorInfo {
                    message: format!("Inference error: {}", e),
                    error_type: "internal_error".to_string(),
                    code: Some("inference_error".to_string()),
                },
            })),
        }
    }
}

pub async fn create_chat_completion(
    request: web::Json<ChatCompletionRequest>,
    data: web::Data<AppState>,
) -> ActixResult<HttpResponse> {
    let stream_requested = request.stream.unwrap_or(false);
    let model_name = request.model.clone();

    // 转换消息格式
    let messages: Vec<Message> = request
        .messages
        .iter()
        .map(|msg| Message {
            role: msg.role.clone(),
            content: Some(msg.content.clone()),
            tool_calls: None,
            name: None,
        })
        .collect();

    // 转换参数到引擎配置
    let engine_config = EngineConfig {
        n_ctx: 4096,
        n_len: request.max_tokens,
        temperature: request.temperature.unwrap_or(0.8),
        top_k: 40,
        top_p: request.top_p.unwrap_or(1.9),
        repeat_penalty: 1.1,
    };

    let engine_arc = match data.model_pool.get_model(&model_name).await {
        Ok(engine) => engine,
        Err(e) => {
            return Ok(HttpResponse::BadRequest().json(ErrorResponse {
                error: ErrorInfo {
                    message: format!("Model not found: {}", e),
                    error_type: "invalid_request_error".to_string(),
                    code: Some("model_not_found".to_string()),
                },
            }));
        }
    };

    // 渲染聊天模板
    let prompt = match crate::template::render_template(
        &engine_arc.get_model_info(),
        &engine_arc
            .get_model_info()
            .template
            .unwrap_or(crate::template::get_default_template()),
        &crate::template::TemplateData::new().with_messages(Some(messages)),
    ) {
        Ok(prompt) => prompt,
        Err(e) => {
            return Ok(HttpResponse::BadRequest().json(ErrorResponse {
                error: ErrorInfo {
                    message: format!("Template rendering error: {}", e),
                    error_type: "invalid_request_error".to_string(),
                    code: Some("template_error".to_string()),
                },
            }));
        }
    };

    if stream_requested {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<StreamChatCompletionResponse>();
        let prompt_clone = prompt.clone();
        let model_name_clone = model_name.clone();
        let engine_arc_clone = engine_arc.clone();

        tokio::task::spawn_blocking(move || {
            let tx_tokens = tx.clone();
            let model_name_clone2 = model_name_clone.clone();
            let request_id = Uuid::new_v4().to_string();
            let request_id_clone = request_id.clone();
            let created = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs();

            // 执行推理并流式发送响应
            let result = engine_arc_clone.infer(
                &prompt_clone,
                Some(&engine_config),
                Some(Box::new(move |tok| {
                    let response = StreamChatCompletionResponse {
                        id: request_id.clone(),
                        object: "chat.completion.chunk".to_string(),
                        created,
                        model: model_name_clone.clone(),
                        choices: vec![StreamChatCompletionChoice {
                            index: 0,
                            delta: ChatMessage {
                                role: "assistant".to_string(),
                                content: tok.into(),
                                name: None,
                            },
                            finish_reason: None,
                        }],
                    };
                    let result = tx_tokens.send(response);
                    if result.is_err() {
                        println!("Error sending response: {:?}", result.err());
                        return false;
                    }
                    true
                })),
            );
            if result.is_err() {
                println!("Error inferring: {:?}", result.err());
                return;
            }

            // 发送结束信号
            let _ = tx.send(StreamChatCompletionResponse {
                id: request_id_clone,
                object: "chat.completion.chunk".to_string(),
                created,
                model: model_name_clone2,
                choices: vec![StreamChatCompletionChoice {
                    index: 0,
                    delta: ChatMessage {
                        role: "assistant".to_string(),
                        content: String::new(),
                        name: None,
                    },
                    finish_reason: Some("stop".to_string()),
                }],
            });
        });

        let stream = tokio_stream::wrappers::UnboundedReceiverStream::new(rx);
        let stream = stream.map(|chunk| {
            let json_str = serde_json::to_string(&chunk).unwrap();
            Ok::<Bytes, actix_web::Error>(Bytes::from(format!("data: {}\n\n", json_str)))
        });

        Ok(HttpResponse::Ok()
            .append_header(("Content-Type", "text/event-stream"))
            .append_header(("Cache-Control", "no-cache"))
            .append_header(("Access-Control-Allow-Origin", "*"))
            .streaming(stream))
    } else {
        match engine_arc.infer(&prompt, Some(&engine_config), None) {
            Ok(text) => {
                let response = ChatCompletionResponse {
                    id: Uuid::new_v4().to_string(),
                    object: "chat.completion".to_string(),
                    created: SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .unwrap()
                        .as_secs(),
                    model: model_name,
                    choices: vec![ChatCompletionChoice {
                        index: 0,
                        message: ChatMessage {
                            role: "assistant".to_string(),
                            content: text,
                            name: None,
                        },
                        finish_reason: "stop".to_string(),
                    }],
                    usage: Usage {
                        prompt_tokens: 0,
                        completion_tokens: 0,
                        total_tokens: 0,
                    },
                };
                Ok(HttpResponse::Ok().json(response))
            }
            Err(e) => Ok(HttpResponse::InternalServerError().json(ErrorResponse {
                error: ErrorInfo {
                    message: format!("Inference error: {}", e),
                    error_type: "internal_error".to_string(),
                    code: Some("inference_error".to_string()),
                },
            })),
        }
    }
}

// 健康检查端点
pub async fn health_check() -> ActixResult<HttpResponse> {
    Ok(HttpResponse::Ok().json(json!({
        "status": "ok",
        "version": env!("CARGO_PKG_VERSION")
    })))
}

// 配置路由
pub fn openai_config(cfg: &mut web::ServiceConfig) {
    cfg.service(
        web::scope("/v1")
            .route("/models", web::get().to(list_models))
            .route("/completions", web::post().to(create_completion))
            .route("/chat/completions", web::post().to(create_chat_completion))
            .route("/health", web::get().to(health_check)),
    );
}
