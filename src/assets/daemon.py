# llm_daemon.py

from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
import threading
import json
import sys
from typing import Dict, Optional
import traceback

# ==============================
# 模型缓存类
# ==============================

stdout_lock = threading.Lock()


class ModelCache:
    _cache: Dict[str, Dict] = {}

    @classmethod
    def get(cls, model_name: str) -> Optional[Dict]:
        if model_name not in cls._cache:
            try:
                print(f"[加载模型] {model_name}", file=sys.stderr)
                model = AutoModelForCausalLM.from_pretrained(
                    model_name, device_map="auto", torch_dtype="auto"
                )
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                cls._cache[model_name] = {
                    "model": model,
                    "tokenizer": tokenizer,
                }
                print(f"[成功加载] {model_name}", file=sys.stderr)
            except Exception as e:
                print(f"[错误] 加载模型失败 {model_name}: {str(e)}", file=sys.stderr)
                return None
        return cls._cache[model_name]

    @classmethod
    def remove(cls, model_name: str):
        if model_name in cls._cache:
            del cls._cache[model_name]


# ==============================
# 流式生成函数（每个请求一个线程）
# ==============================


def stream_generation(req_id: str, model_name: str, prompt: str, max_tokens: int):
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

        # 启动生成线程
        def generate():
            try:
                model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    streamer=streamer,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                )
            except Exception as e:
                print(f"[生成错误] {req_id}: {e}", file=sys.stderr)
            finally:
                streamer.end()  # 必须调用

        thread = threading.Thread(target=generate)
        thread.start()

        # 实时发送 token
        buffer = ""
        for new_text in streamer:
            if new_text:
                # 尝试减少碎片（可选合并）
                buffer += new_text
                # 分块发送（可根据需要调整策略）
                with stdout_lock:
                    print(json.dumps({"req_id": req_id, "token": new_text}))
                    sys.stdout.flush()

        thread.join(timeout=2)
        if thread.is_alive():
            print(f"[警告] 生成线程未正常退出: {req_id}", file=sys.stderr)

        # 完成信号
        with stdout_lock:
            print(json.dumps({"req_id": req_id, "done": True}))
            sys.stdout.flush()

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

print("[系统] LLM Daemon 启动，等待请求...", file=sys.stderr)
sys.stderr.flush()

try:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            request = json.loads(line)
            req_id = request.get("req_id", "unknown")
            model_name = request.get("model", "Qwen/Qwen3-0.6B")
            prompt = request.get("prompt", "")
            max_tokens = request.get("max_new_tokens", 256)

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

            # 启动独立线程处理请求（支持并发）
            t = threading.Thread(
                target=stream_generation,
                kwargs={
                    "req_id": req_id,
                    "model_name": model_name,
                    "prompt": prompt,
                    "max_tokens": max_tokens,
                },
            )
            t.start()

        except json.JSONDecodeError as e:
            with stdout_lock:
                print(
                    json.dumps({"req_id": "system", "error": f"Invalid JSON: {str(e)}"})
                )
                sys.stdout.flush()

except (KeyboardInterrupt, EOFError):
    print("[系统] 收到退出信号，关闭 daemon...", file=sys.stderr)
    sys.stderr.flush()
