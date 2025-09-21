use super::pool::ModelPool;
use std::{error::Error, sync::Arc};

use actix_web::{App, HttpServer, web};

#[actix_web::get("/rllama/load/{model_name}")]
async fn load_model(
    path: web::Path<String>,
    data: web::Data<AppState>,
) -> Result<String, Box<dyn Error>> {
    let model_name = path.into_inner();
    data.model_pool.get_model(&model_name).await?;
    println!("[API] Model '{}' loaded.", model_name);
    Ok(format!("Model '{}' loaded.", model_name))
}

// AppState 结构体，用于在 Actix-web 应用中共享状态
struct AppState {
    model_pool: Arc<ModelPool>,
}

// 启动 API 服务器的入口函数
pub async fn start_api_server(host: String, port: u16) -> std::io::Result<()> {
    let model_pool = Arc::new(ModelPool::new());

    println!("Server starting on http://{}:{}/", host, port);
    HttpServer::new(move || {
        // 每个 worker 线程都会克隆一份 Arc<ModelPool>
        App::new()
            // 使用 .app_data() 将共享状态添加到 Actix-web 应用
            .app_data(web::Data::new(AppState {
                model_pool: Arc::clone(&model_pool), // 为每个 worker 线程克隆 Arc
            }))
            // 另一个示例路由
            .route(
                "/status",
                web::get().to(|| async { "API Server is running!" }),
            )
            .service(load_model)
    })
    .bind((host, port))?
    .run()
    .await?;
    Ok(())
}
