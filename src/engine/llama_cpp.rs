use crate::cli::Cli;
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::ggml_time_us;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel, Special};
use llama_cpp_2::sampling::LlamaSampler;

use std::io::Write;
use std::num::NonZeroU32;
use std::time::Duration;

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
    pub fn new(args: &Cli) -> Result<Self, Box<dyn std::error::Error>> {
        let backend = LlamaBackend::init()?;
        let model_params = LlamaModelParams::default();
        let model = LlamaModel::load_from_file(&backend, &args.model_path, &model_params)?;
        
        let n_ctx = args.n_ctx.unwrap_or(2048);
        let n_len = args.n_len.unwrap_or(1024);
        let temperature = args.temperature.unwrap_or(0.7);
        let top_k = args.top_k.unwrap_or(40);
        let top_p = args.top_p.unwrap_or(0.9);
        let repeat_penalty = args.repeat_penalty.unwrap_or(1.1);

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

    fn make_prompt(prompt: &str) -> String {
        format!(
            "user\n{}
assistant\n",
            prompt,
        )
    }
}

use crate::engine::InferenceEngine;

impl InferenceEngine for LlamaEngine {
    fn infer(&mut self, prompt: &str) -> Result<String, Box<dyn std::error::Error>> {
        let prompt_text = Self::make_prompt(prompt);
        
        // 设置上下文参数
        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(Some(NonZeroU32::new(self.n_ctx as u32).unwrap()))
            .with_n_batch(2048)
            .with_n_ubatch(512);

        // 创建上下文
        let mut ctx = self.model.new_context(&self.backend, ctx_params)?;

        // Tokenize 提示
        let tokens_list = self.model.str_to_token(&prompt_text, AddBos::Always)?;

        // Create batch with explicit logits configuration for the initial prompt
        // 初始批处理，只对最后一个 token 请求 logits
        let mut batch = LlamaBatch::new(tokens_list.len(), 1); // 容量为提示 token 数量，批次大小为 1
        for (i, &token) in tokens_list.iter().enumerate() {
            let logits = i == tokens_list.len() - 1; // 只有最后一个 token 需要 logits
            batch.add(token, i as i32, &[0], logits)?;
        }

        // 解码初始提示
        ctx.decode(&mut batch)?;

        let mut output = String::new();

        // 采样器
        let mut sampler = LlamaSampler::chain_simple([
            LlamaSampler::temp(self.temperature),
            LlamaSampler::top_p(self.top_p, 1),
            LlamaSampler::top_k(self.top_k),
            LlamaSampler::penalties(64, self.repeat_penalty, 0.0, 0.0),
            LlamaSampler::greedy(),
        ])
        .with_tokens(tokens_list.iter().copied()); // 将所有提示 token 添加到采样器历史

        let mut n_cur = batch.n_tokens(); // 当前上下文中 token 的数量
        let mut n_decode = 0; // 实际解码的新 token 数量

        // 主生成循环
        while n_cur < self.n_ctx && n_decode < self.n_len as i32 {
            // 采样下一个 token
            let token = sampler.sample(&ctx, batch.n_tokens() - 1); // 总是从最后一个已解码 token 的 logits 采样

            // 检查是否是 EOS (End Of Stream) token
            if self.model.is_eog_token(token) {
                break;
            }

            // 将 token 转换为字符串并添加到输出
            let token_str = self.model.token_to_str(token, Special::Plaintext)?;
            output.push_str(&token_str);
            
            // 将新生成的 token 添加到采样器历史中，以便后续惩罚计算
            sampler.accept(token);

            // 清空批次并添加新生成的 token
            batch.clear();
            // 新生成的 token 的 pos 应该是 n_cur
            batch.add(token, n_cur as i32, &[0], true)?; // 只添加一个 token，并请求其 logits

            n_cur += 1;
            n_decode += 1; // 统计生成的新 token 数量

            // 解码新批次
            ctx.decode(&mut batch)?;
        }

        Ok(output)
    }
}