use lazy_static::lazy_static;

#[cfg(feature = "llama-cpp-2")]
use llama_cpp_2::context::params::KvCacheType;

lazy_static! {
    pub static ref TLLAMA_THREADS: i32 = std::env::var("TLLAMA_THREADS")
        .map(|s| s.parse::<i32>().unwrap())
        .unwrap_or(
            std::thread::available_parallelism()
                .map(|n| n.get() as i32)
                .unwrap_or(4)
        );
    pub static ref TLLAMA_MODEL_PATHS: Vec<String> = std::env::var("TLLAMA_MODEL_PATHS")
        .map(|s| { s.split(",").map(|s| s.to_string()).collect() })
        .unwrap_or(vec![]);

    #[cfg(feature = "llama-cpp-2")]
    pub static ref TLLAMA_FLASH_ATTN: i32 = std::env::var("TLLAMA_FLASH_ATTN")
        .map(|s| s.parse::<i32>().unwrap())
        .map(|s| {
            if s < -1 || s > 1 {
                panic!("TLLAMA_FLASH_ATTN must be -1, 0, 1");
            }
            s
        })
        .unwrap_or(-1);

    #[cfg(feature = "llama-cpp-2")]
    pub static ref TLLAMA_KV_CACHE_TYPE: KvCacheType =
        std::env::var("TLLAMA_KV_CACHE_TYPE").map(|s| match s.to_lowercase().as_str() {
            "f32" => KvCacheType::F32,
            "f16" => KvCacheType::F16,
            "q4_0" => KvCacheType::Q4_0,
            "q4_1" => KvCacheType::Q4_1,
            "q5_0" => KvCacheType::Q5_0,
            "q5_1" => KvCacheType::Q5_1,
            "q8_0" => KvCacheType::Q8_0,
            "q8_1" => KvCacheType::Q8_1,
            "q2_k" => KvCacheType::Q2_K,
            "q3_k" => KvCacheType::Q3_K,
            "q4_k" => KvCacheType::Q4_K,
            "q5_k" => KvCacheType::Q5_K,
            "q6_k" => KvCacheType::Q6_K,
            "q8_k" => KvCacheType::Q8_K,
            "iq2_xxs" => KvCacheType::IQ2_XXS,
            "iq2_xs" => KvCacheType::IQ2_XS,
            "iq3_xxs" => KvCacheType::IQ3_XXS,
            "iq1_s" => KvCacheType::IQ1_S,
            "iq4_nl" => KvCacheType::IQ4_NL,
            "iq3_s" => KvCacheType::IQ3_S,
            "iq2_s" => KvCacheType::IQ2_S,
            "iq4_xs" => KvCacheType::IQ4_XS,
            "i8" => KvCacheType::I8,
            "i16" => KvCacheType::I16,
            "i32" => KvCacheType::I32,
            "i64" => KvCacheType::I64,
            "f64" => KvCacheType::F64,
            "iq1_m" => KvCacheType::IQ1_M,
            "bf16" => KvCacheType::BF16,
            "tq1_0" => KvCacheType::TQ1_0,
            "tq2_0" => KvCacheType::TQ2_0,
            "mxfp4" => KvCacheType::MXFP4,
            _ => KvCacheType::F16
        }).unwrap_or(KvCacheType::F16);

        pub static ref TLLAMA_RPC_HOST: String = std::env::var("TLLAMA_RPC_HOST")
        .unwrap_or("http://127.0.0.1:12186".to_string());
}
