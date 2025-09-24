use super::pool::ModelPool;
use std::sync::Arc;

use actix_web::{App, HttpServer, web};

pub struct AppState {
    pub model_pool: Arc<ModelPool>,
}

// 启动 API 服务器的入口函数
pub async fn start_api_server(host: String, port: u16) -> std::io::Result<()> {
    let model_pool = Arc::new(ModelPool::new());

    println!("Server starting on http://{}:{}/", host, port);
    HttpServer::new(move || {
        App::new()
            .app_data(web::Data::new(AppState {
                model_pool: Arc::clone(&model_pool),
            }))
            .configure(super::rllama_api::rllama_config)
            .configure(super::openai::openai_config)
    })
    .bind((host, port))?
    .run()
    .await?;
    Ok(())
}
