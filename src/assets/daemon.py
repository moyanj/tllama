from email import message
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
import torch
import time
import threading


def stream_generation(model, tokenizer, prompt, streamer, max_length=100):
    """流式生成器函数"""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]
    inputs = tokenizer.apply_chat_template(messages, return_tensors="pt")
    print(inputs)
    # 配置生成参数
    generation_kwargs = {
        "input_ids": inputs,
        "max_new_tokens": max_length,
        "streamer": streamer,
        "do_sample": True,
        "temperature": 0.7,
    }

    # 逐步生成
    model.generate(**generation_kwargs)


# 使用流式生成器
model_name = "Qwen/Qwen3-0.6B"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

"""
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
"""
prompt = "机器学习的应用包括"

print("开始流式生成:")

streamer = TextIteratorStreamer(tokenizer, skip_prompt=True)
cfg = {
    "model": model,
    "tokenizer": tokenizer,
    "prompt": prompt,
    "streamer": streamer,
}
thread = threading.Thread(target=stream_generation, kwargs=cfg)
thread.start()

for new_text in streamer:
    print(new_text, end="", flush=True)

thread.join()
