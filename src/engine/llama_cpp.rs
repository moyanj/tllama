use crate::discover::Model;
use crate::engine::{EngineBackend, EngineCallback, EngineConfig};
use anyhow::Result;
use lazy_static::lazy_static;
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel, Special};
use llama_cpp_2::sampling::LlamaSampler;
use std::num::NonZeroU32;

lazy_static! {
    pub static ref LLAMA_BACKEND: LlamaBackend = LlamaBackend::init().unwrap();
}

// 声明LlamaEngine是线程安全的
unsafe impl Send for LlamaEngine {}
unsafe impl Sync for LlamaEngine {}

pub struct LlamaEngine {
    model_info: Model,
    model: LlamaModel,
    args: EngineConfig,
}

impl EngineBackend for LlamaEngine {
    fn new(args: &EngineConfig, model_info: &Model) -> Result<Self> {
        let model_params = LlamaModelParams::default();
        let model = LlamaModel::load_from_file(&LLAMA_BACKEND, &model_info.path, &model_params)?;

        Ok(LlamaEngine {
            model,
            args: (*args).clone(),
            model_info: model_info.clone(),
        })
    }
    fn get_model_info(&self) -> Model {
        self.model_info.clone()
    }

    fn infer(
        &self,
        prompt: &str,
        args: Option<&EngineConfig>,
        mut callback: Option<EngineCallback>,
    ) -> Result<String> {
        // 获取EngineConfig实例
        let args = args.unwrap_or(&self.args);
        let mut decoder = encoding_rs::UTF_8.new_decoder();
        // 设置上下文参数
        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(Some(NonZeroU32::new(args.n_ctx as u32).unwrap()))
            .with_n_batch(2048)
            .with_n_ubatch(512)
            .with_n_threads(*crate::env::TLLAMA_THREADS)
            .with_n_threads_batch(*crate::env::TLLAMA_THREADS)
            .with_flash_attention(*crate::env::TLLAMA_FLASH_ATTN);
        // 创建上下文
        let mut ctx = self.model.new_context(&LLAMA_BACKEND, ctx_params)?;
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
            LlamaSampler::temp(args.temperature),
            LlamaSampler::top_p(args.top_p, 1),
            LlamaSampler::top_k(args.top_k),
            LlamaSampler::penalties(64, args.repeat_penalty, 0.0, 0.0),
            LlamaSampler::greedy(),
        ])
        .with_tokens(tokens_list.iter().copied());
        let mut n_cur = batch.n_tokens();
        let mut n_decode = 0;
        let mut output = String::new();

        let max_tokens = args.n_len.map(|n| n as i32);
        while max_tokens.map_or(true, |max| n_decode < max) {
            // 采样下一个token
            let token = sampler.sample(&ctx, -1);
            // 检查是否是EOS
            if self.model.is_eog_token(token) {
                break;
            }

            // 将token转换为字符串并输出
            let output_bytes = self.model.token_to_bytes(token, Special::Tokenize)?;
            // use `Decoder.decode_to_string()` to avoid the intermediate buffer
            let mut token_str = String::with_capacity(32);
            let _decode_result = decoder.decode_to_string(&output_bytes, &mut token_str, false);

            // 调用回调函数处理输出
            if callback.is_some() {
                let shoud_stop = callback.as_mut().unwrap()(token_str.clone());
                if shoud_stop {
                    break;
                }
            }

            // 将新生成的token添加到采样器历史中
            sampler.accept(token);
            // 清空批次并添加新生成的token
            batch.clear();
            batch.add(token, n_cur as i32, &[0], true)?;
            n_cur += 1;
            n_decode += 1;
            output += &token_str;
            // 解码新批次
            ctx.decode(&mut batch)?;
        }
        Ok(output)
    }
}
