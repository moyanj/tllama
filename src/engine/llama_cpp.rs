use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel, Special};
use llama_cpp_2::sampling::LlamaSampler;

use crate::engine::{EngineConfig, InferenceEngine};
use std::num::NonZeroU32;

pub struct LlamaEngine {
    model: LlamaModel,
    backend: LlamaBackend,
    n_ctx: i32,
    n_len: u32,
    temperature: f32,
    top_k: i32,
    top_p: f32,
    repeat_penalty: f32,
}

impl LlamaEngine {
    pub fn new(
        args: &EngineConfig,
        model_path: &String,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let backend = LlamaBackend::init()?;
        let model_params = LlamaModelParams::default();
        let model = LlamaModel::load_from_file(&backend, model_path, &model_params)?;

        let n_ctx = args.n_ctx;
        let n_len = args.n_len;
        let temperature = args.temperature;
        let top_k = args.top_k;
        let top_p = args.top_p;
        let repeat_penalty = args.repeat_penalty;

        Ok(LlamaEngine {
            model,
            backend,
            n_ctx,
            n_len,
            temperature,
            top_k,
            top_p,
            repeat_penalty,
        })
    }
}

impl InferenceEngine for LlamaEngine {
    fn infer(&mut self, prompt: &str) -> Result<String, Box<dyn std::error::Error>> {
        let mut output = String::new();
        self.infer_stream(prompt, |token| {
            output.push_str(token);
            Ok(())
        })?;
        Ok(output)
    }

    fn infer_stream(
        &mut self,
        prompt: &str,
        mut callback: impl FnMut(&str) -> Result<(), Box<dyn std::error::Error>>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // 设置上下文参数
        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(Some(NonZeroU32::new(self.n_ctx as u32).unwrap()))
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
        // 采样器 - 移除了greedy()以支持随机采样
        let mut sampler = LlamaSampler::chain_simple([
            LlamaSampler::temp(self.temperature),
            LlamaSampler::top_p(self.top_p, 1),
            LlamaSampler::top_k(self.top_k),
            LlamaSampler::penalties(64, self.repeat_penalty, 0.0, 0.0),
            LlamaSampler::greedy(),
        ])
        .with_tokens(tokens_list.iter().copied());
        let mut n_cur = batch.n_tokens();
        let mut n_decode = 0;
        // 主生成循环
        while n_cur < self.n_ctx && n_decode < self.n_len as i32 {
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
}
