from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
import threading
import json
import sys
from typing import Dict, Optional
import traceback
from dataclasses import dataclass
import os

# ==============================
# 模型缓存类
# ==============================

stdout_lock = threading.Lock()


def log(msg: str):
    if os.environ.get("TLLAMA_DAEMON"):
        return
    with stdout_lock:
        print(msg, file=sys.stderr)
        sys.stdout.flush()


@dataclass
class Args:
    n_ctx: int = 2048
    n_len: Optional[int] = None
    temperature: float = 0.7
    top_k: int = 40
    top_p: float = 0.9
    repeat_penalty: float = 1.1


class ModelCache:
    _cache: Dict[str, Dict] = {}

    @classmethod
    def get(cls, model_name: str) -> Optional[Dict]:
        if model_name not in cls._cache:
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    model_name, device_map="auto", dtype="auto"
                )
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                cls._cache[model_name] = {
                    "model": model,
                    "tokenizer": tokenizer,
                }
            except Exception as e:
                log(f"[错误] 加载模型失败 {model_name}: {str(e)}")
                return None
        return cls._cache[model_name]

    @classmethod
    def remove(cls, model_name: str):
        if model_name in cls._cache:
            del cls._cache[model_name]


# ==============================
# 流式生成函数（每个请求一个线程）
# ==============================


def stream_generation(req_id: str, model_name: str, prompt: str, args: dict):
    try:
        # 获取模型
        model_entry = ModelCache.get(model_name)
        if not model_entry:
            with stdout_lock:
                print(
                    json.dumps(
                        {"req_id": req_id, "error": f"Model '{model_name}' not found"}
                    )
                )
                sys.stdout.flush()
            return

        model = model_entry["model"]
        tokenizer = model_entry["tokenizer"]

        # 编码输入
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        streamer = TextIteratorStreamer(
            tokenizer, skip_prompt=True, skip_special_tokens=True
        )

        # 设置生成参数
        generation_config = {
            "max_length": args.get("n_ctx", 4096),
            "max_new_tokens": args.get("n_len", 1e30) or 1e30,  # 默认值256
            "temperature": args.get("temperature", 0.7),
            "top_k": args.get("top_k", 40),
            "top_p": args.get("top_p", 0.9),
            "repetition_penalty": args.get("repeat_penalty", 1.1),
            "do_sample": True,
            "streamer": streamer,
        }

        if generation_config["max_length"] < generation_config["max_new_tokens"]:
            del generation_config["max_new_tokens"]

        # 启动生成线程
        def generate():
            try:
                model.generate(**inputs, **generation_config)
            except Exception as e:
                log(f"[生成错误] {req_id}: {e}")
            finally:
                streamer.end()  # 必须调用
            return

        thread = threading.Thread(target=generate)
        thread.start()

        # 实时发送 token
        for new_text in streamer:
            if new_text:
                with stdout_lock:
                    print(
                        json.dumps(
                            {"req_id": req_id, "token": new_text}, ensure_ascii=False
                        )
                    )
                    sys.stdout.flush()

        # 完成信号
        with stdout_lock:
            print(json.dumps({"req_id": req_id, "done": True}))
            sys.stdout.flush()

        thread.join(timeout=2)

    except Exception as e:
        error_tb = traceback.format_exc()
        with stdout_lock:
            print(
                json.dumps({"req_id": req_id, "error": str(e), "traceback": error_tb})
            )
            sys.stdout.flush()


# ==============================
# 主循环：读取 stdin JSON 请求
# ==============================

sys.stderr.flush()
try:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
            req_id = request.get("req_id", "unknown")
            # 判断是否为控制命令：load / unload
            cmd = request.get("cmd")
            if cmd == "load":
                model_name = request.get("model")
                if not model_name:
                    with stdout_lock:
                        print(
                            json.dumps(
                                {"req_id": req_id, "error": "Missing 'model' field"}
                            )
                        )
                        sys.stdout.flush()
                    continue
                model_entry = ModelCache.get(model_name)
                if model_entry:
                    with stdout_lock:
                        print(
                            json.dumps(
                                {"req_id": req_id, "loaded": True, "model": model_name}
                            )
                        )
                        sys.stdout.flush()
                else:
                    with stdout_lock:
                        print(
                            json.dumps(
                                {
                                    "req_id": req_id,
                                    "loaded": False,
                                    "model": model_name,
                                    "error": "Failed to load model",
                                }
                            )
                        )
                        sys.stdout.flush()
                continue
            elif cmd == "unload":
                model_name = request.get("model")
                if not model_name:
                    with stdout_lock:
                        print(
                            json.dumps(
                                {"req_id": req_id, "error": "Missing 'model' field"}
                            )
                        )
                        sys.stdout.flush()
                    continue
                if model_name in ModelCache._cache:
                    # 尝试清理显存
                    ModelCache.remove(model_name)
                    # 释放缓存
                    import torch

                    torch.cuda.empty_cache()
                    with stdout_lock:
                        print(
                            json.dumps(
                                {
                                    "req_id": req_id,
                                    "unloaded": True,
                                    "model": model_name,
                                }
                            )
                        )
                        sys.stdout.flush()
                else:
                    with stdout_lock:
                        print(
                            json.dumps(
                                {
                                    "req_id": req_id,
                                    "unloaded": False,
                                    "model": model_name,
                                    "error": "Model not loaded",
                                }
                            )
                        )
                        sys.stdout.flush()
                continue
            elif cmd == "exit":
                sys.exit(0)
            # 如果不是命令，则视为生成请求
            model_name = request.get("model", "Qwen/Qwen3-0.6B")
            prompt = request.get("prompt", "")
            args = request.get("args", {})
            if not prompt:
                with stdout_lock:
                    print(
                        json.dumps(
                            {
                                "req_id": req_id,
                                "error": "Empty prompt",
                                "prompt": prompt,
                            }
                        )
                    )
                    sys.stdout.flush()
                continue
            # 启动独立线程处理生成请求
            t = threading.Thread(
                target=stream_generation,
                kwargs={
                    "req_id": req_id,
                    "model_name": model_name,
                    "prompt": prompt,
                    "args": args,
                },
            )
            t.start()
        except json.JSONDecodeError as e:
            with stdout_lock:
                print(
                    json.dumps({"req_id": "system", "error": f"Invalid JSON: {str(e)}"})
                )
                sys.stdout.flush()
except KeyboardInterrupt:
    pass
