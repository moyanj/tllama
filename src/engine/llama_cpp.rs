use crate::discover::Model;
use crate::engine::{EngineConfig, InferenceEngine};
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel, Special};
use llama_cpp_2::sampling::LlamaSampler;
use std::num::NonZeroU32;

// 声明LlamaEngine是线程安全的
unsafe impl Send for LlamaEngine {}
unsafe impl Sync for LlamaEngine {}

pub struct LlamaEngine {
    model_info: Model,
    model: LlamaModel,
    backend: LlamaBackend,
    args: EngineConfig,
}

impl LlamaEngine {
    pub fn new(
        args: &EngineConfig,
        model_info: &Model,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        llama_cpp_2::send_logs_to_tracing(
            llama_cpp_2::LogOptions::default().with_logs_enabled(false),
        );
        let backend = LlamaBackend::init()?;
        let model_params = LlamaModelParams::default();
        let model = LlamaModel::load_from_file(&backend, &model_info.model_path, &model_params)?;

        Ok(LlamaEngine {
            model,
            backend,
            args: (*args).clone(),
            model_info: model_info.clone(),
        })
    }
}

impl InferenceEngine for LlamaEngine {
    fn get_model_info(&self) -> Model {
        self.model_info.clone()
    }
    fn infer(&mut self, prompt: &str) -> Result<String, Box<dyn std::error::Error>> {
        let mut output = String::new();
        self.infer_stream(prompt, &mut |token| {
            output.push_str(token);
            Ok(())
        })?;
        Ok(output)
    }

    fn infer_stream(
        &mut self,
        prompt: &str,
        callback: &mut dyn FnMut(&str) -> Result<(), Box<dyn std::error::Error>>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // 设置上下文参数
        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(Some(NonZeroU32::new(self.args.n_ctx as u32).unwrap()))
            .with_n_batch(2048)
            .with_n_ubatch(512)
            .with_n_threads(20);
        // 创建上下文
        let mut ctx = self.model.new_context(&self.backend, ctx_params)?;
        // Tokenize提示
        let tokens_list = self.model.str_to_token(&prompt, AddBos::Always)?;
        // 创建初始batch
        let mut batch = LlamaBatch::new(tokens_list.len(), 1);
        for (i, &token) in tokens_list.iter().enumerate() {
            let logits = i == tokens_list.len() - 1;
            batch.add(token, i as i32, &[0], logits)?;
        }
        // 解码初始提示
        ctx.decode(&mut batch)?;

        let mut sampler = LlamaSampler::chain_simple([
            LlamaSampler::temp(self.args.temperature),
            LlamaSampler::top_p(self.args.top_p, 1),
            LlamaSampler::top_k(self.args.top_k),
            LlamaSampler::penalties(64, self.args.repeat_penalty, 0.0, 0.0),
            LlamaSampler::greedy(),
        ])
        .with_tokens(tokens_list.iter().copied());
        let mut n_cur = batch.n_tokens();
        let mut n_decode = 0;
        // 主生成循环
        while n_cur < self.args.n_ctx && n_decode < self.args.n_len as i32 {
            // 采样下一个token
            let token = sampler.sample(&ctx, batch.n_tokens() - 1);
            // 检查是否是EOS
            if self.model.is_eog_token(token) {
                break;
            }
            // 将token转换为字符串并输出
            let token_str = self.model.token_to_str(token, Special::Plaintext)?;

            // 调用回调函数处理输出
            callback(&token_str)?;
            // 将新生成的token添加到采样器历史中
            sampler.accept(token);
            // 清空批次并添加新生成的token
            batch.clear();
            batch.add(token, n_cur as i32, &[0], true)?;
            n_cur += 1;
            n_decode += 1;
            // 解码新批次
            ctx.decode(&mut batch)?;
        }
        Ok(())
    }
    fn set_config(&mut self, config: &EngineConfig) {
        self.args = (*config).clone();
    }
}
