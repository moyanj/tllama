use super::server::AppState;
use crate::template::Message;
use crate::{discover::MODEL_DISCOVERER, engine::EngineConfig};
use actix_web::web::Bytes;
use actix_web::{HttpResponse, Result as ActixResult, web};
use serde_json::json;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio_stream::StreamExt;

#[derive(serde::Deserialize)]
struct InferArgs {
    pub model: String,               // 模型名称
    pub prompt: String,              // 输入
    pub n_len: Option<u32>,          // 输出长度
    pub temperature: Option<f32>,    // 采样温度
    pub top_k: Option<i32>,          // 采样 top-k
    pub top_p: Option<f32>,          // 采样 top-p
    pub repeat_penalty: Option<f32>, // 重复惩罚
    pub n_ctx: Option<i32>,          // 上下文长度
    pub stream: Option<bool>,        // 是否流式返回
}

#[derive(serde::Deserialize)]
struct ChatArgs {
    pub model: String,               // 模型名称
    pub messages: Vec<Message>,      // 聊天消息
    pub n_len: Option<u32>,          // 输出长度
    pub temperature: Option<f32>,    // 采样温度
    pub top_k: Option<i32>,          // 采样 top-k
    pub top_p: Option<f32>,          // 采样 top-p
    pub repeat_penalty: Option<f32>, // 重复惩罚
    pub n_ctx: Option<i32>,          // 上下文长度
    pub stream: Option<bool>,        // 是否流式返回
}

#[derive(Debug, serde::Serialize)]
struct StreamChunk {
    id: String,
    content: String,
    created: u64,
    model: String,
    finished: bool,
    finish_reason: Option<String>,
}

async fn common_inference(
    model_name: String,
    prompt: String,
    data: web::Data<AppState>,
    stream_requested: bool,
    engine_config: EngineConfig,
) -> ActixResult<HttpResponse> {
    let engine_mutex_arc = match data.model_pool.get_model(&model_name).await {
        Ok(engine) => engine,
        Err(e) => {
            return Ok(HttpResponse::InternalServerError().json(json!({
                "error": e.to_string()
            })));
        }
    };

    // 设置引擎配置
    engine_mutex_arc.lock().await.set_config(&engine_config);

    if stream_requested {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<StreamChunk>();
        let prompt_clone = prompt.clone();
        let model_name_clone = model_name.clone();
        let engine_mutex_arc_clone = engine_mutex_arc.clone();

        tokio::spawn(async move {
            let tx_tokens = tx.clone();
            let model_name_clone2 = model_name_clone.clone();

            // 执行推理并流式发送响应
            let _ = engine_mutex_arc_clone.lock().await.infer(
                &prompt_clone,
                Some(Box::new(move |tok| {
                    let _ = tx_tokens.send(StreamChunk {
                        id: "".into(),
                        content: tok.into(),
                        created: SystemTime::now()
                            .duration_since(UNIX_EPOCH)
                            .unwrap()
                            .as_secs(),
                        model: model_name_clone.clone(),
                        finished: false,
                        finish_reason: None,
                    });
                })),
            );

            // 发送结束信号
            let _ = tx.send(StreamChunk {
                id: "".into(),
                content: "".into(),
                created: SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_secs(),
                model: model_name_clone2,
                finished: true,
                finish_reason: Some("stop".into()),
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
            .streaming(stream))
    } else {
        // 非流式推理
        match engine_mutex_arc.lock().await.infer(&prompt, None) {
            Ok(text) => Ok(HttpResponse::Ok().json(json!({ "response": text }))),
            Err(e) => Ok(HttpResponse::InternalServerError().json(json!({
                "error": e.to_string()
            }))),
        }
    }
}

#[actix_web::get("/rllama/load/{model_name:.*}")]
pub async fn load_model(
    path: web::Path<String>,
    data: web::Data<AppState>,
) -> ActixResult<HttpResponse> {
    let model_name = path.into_inner();
    match data.model_pool.get_model(&model_name).await {
        Ok(engine_mutex_arc) => {
            let engine = engine_mutex_arc.lock().await;
            let model_info = engine.get_model_info();
            println!("[API] Model '{}' loaded.", model_name);
            Ok(HttpResponse::Ok().json(model_info))
        }
        Err(e) => Ok(HttpResponse::InternalServerError().json(json!({
            "error": e.to_string()
        }))),
    }
}

#[actix_web::get("/rllama/unload/{model_name:.*}")]
pub async fn unload_model(
    path: web::Path<String>,
    data: web::Data<AppState>,
) -> ActixResult<HttpResponse> {
    let model_name = path.into_inner();
    data.model_pool.unload_model(&model_name).await;
    Ok(HttpResponse::Ok().json(json!({"message": "Model unloaded."})))
}

#[actix_web::get("/rllama/list")]
pub async fn list_models() -> ActixResult<HttpResponse> {
    let models = match MODEL_DISCOVERER.lock() {
        Ok(discoverer) => discoverer.get_model_list().clone(),
        Err(e) => {
            return Ok(HttpResponse::InternalServerError().json(json!({
                "error": format!("Failed to acquire model discoverer lock: {}", e)
            })));
        }
    };
    Ok(HttpResponse::Ok().json(models))
}

#[actix_web::post("/rllama/infer")]
pub async fn infer(
    args: web::Query<InferArgs>,
    data: web::Data<AppState>,
) -> ActixResult<HttpResponse> {
    let model_name = args.model.clone();
    let prompt = args.prompt.clone();
    let stream_requested = args.stream.unwrap_or(false);

    // 构造配置
    let engine_config = EngineConfig {
        n_ctx: args.n_ctx.unwrap_or(2048),
        n_len: args.n_len.unwrap_or(512),
        temperature: args.temperature.unwrap_or(0.8),
        top_k: args.top_k.unwrap_or(40),
        top_p: args.top_p.unwrap_or(0.95),
        repeat_penalty: args.repeat_penalty.unwrap_or(1.1),
    };

    common_inference(model_name, prompt, data, stream_requested, engine_config).await
}

#[actix_web::post("/rllama/chat")]
pub async fn chat(
    args: web::Query<ChatArgs>,
    data: web::Data<AppState>,
) -> ActixResult<HttpResponse> {
    let model_name = args.model.clone();
    let stream_requested = args.stream.unwrap_or(false);

    let engine_config = EngineConfig {
        n_ctx: args.n_ctx.unwrap_or(2048),
        n_len: args.n_len.unwrap_or(512),
        temperature: args.temperature.unwrap_or(0.8),
        top_k: args.top_k.unwrap_or(40),
        top_p: args.top_p.unwrap_or(0.95),
        repeat_penalty: args.repeat_penalty.unwrap_or(1.1),
    };

    let prompt = crate::template::render_chatml_template(&crate::template::PromptData {
        system: None,
        messages: Some(args.messages.clone()),
        tools: None,
        prompt: None,
        response: None,
    })?;

    common_inference(model_name, prompt, data, stream_requested, engine_config).await
}

#[actix_web::get("/rllama/discover")]
pub async fn discover() -> ActixResult<HttpResponse> {
    let models = match MODEL_DISCOVERER.lock() {
        Ok(discoverer) => discoverer.get_model_list().clone(),
        Err(e) => {
            return Ok(HttpResponse::InternalServerError().json(json!({
                "error": format!("Failed to acquire model discoverer lock: {}", e)
            })));
        }
    };
    Ok(HttpResponse::Ok().json(models))
}

pub fn rllama_config(cfg: &mut web::ServiceConfig) {
    cfg.service(load_model)
        .service(unload_model)
        .service(list_models)
        .service(infer)
        .service(chat)
        .service(discover);
}
