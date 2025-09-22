use super::server::AppState;
use crate::{discover::MODEL_DISCOVERER, engine::EngineConfig};
use actix_web::web::Bytes;
use actix_web::{HttpResponse, Result as ActixResult, web};
use futures::stream::Stream;
use serde_json::json;
use std::pin::Pin;
use std::task::{Context, Poll};
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::mpsc;
use tokio_stream::StreamExt;
use tokio_stream::wrappers::ReceiverStream;

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

#[derive(Debug, serde::Serialize)]
struct StreamChunk {
    id: String,
    content: String,
    created: u64,
    model: String,
    finished: bool,
    finish_reason: Option<String>,
}

// 自定义流类型包装器
struct ChunkStream {
    inner: ReceiverStream<Result<Bytes, actix_web::Error>>,
}

impl Stream for ChunkStream {
    type Item = Result<Bytes, actix_web::Error>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        Pin::new(&mut self.inner).poll_next(cx)
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

    // 获取模型引擎实例
    let engine_mutex_arc = match data.model_pool.get_model(&model_name).await {
        Ok(engine_mutex_arc) => engine_mutex_arc,
        Err(e) => {
            return Ok(HttpResponse::InternalServerError().json(json!({
                "error": e.to_string()
            })));
        }
    };

    // 应用引擎配置
    let engine_config = EngineConfig {
        n_ctx: args.n_ctx.unwrap_or(2048),
        n_len: args.n_len.unwrap_or(512),
        temperature: args.temperature.unwrap_or(0.8),
        top_k: args.top_k.unwrap_or(40),
        top_p: args.top_p.unwrap_or(0.95),
        repeat_penalty: args.repeat_penalty.unwrap_or(1.1),
    };

    engine_mutex_arc.lock().await.set_config(&engine_config);

    if stream_requested {
        Ok(HttpResponse::Ok()
            .json(json!({"error": "Streaming not supported for non-streaming inference."})))
    } else {
        // 非流式推理（保持不变）
        let response = engine_mutex_arc.lock().await.infer(&prompt);
        match response {
            Ok(text) => Ok(HttpResponse::Ok().json(json!({ "response": text }))),
            Err(e) => Ok(HttpResponse::InternalServerError().json(json!({
                "error": e.to_string()
            }))),
        }
    }
}

pub fn rllama_config(cfg: &mut web::ServiceConfig) {
    cfg.service(load_model)
        .service(unload_model)
        .service(list_models)
        .service(infer);
}
