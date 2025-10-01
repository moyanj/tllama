# 🧠 Tllama
**🚀 Lightweight Local LLM Inference Engine**

Tllama is a Rust-based open-source LLM engine designed for efficient local execution. It features a **command-line interface** and **OpenAI-compatible API** for seamless model interaction.

---

## 🚀 Key Features
- 🔍 Smart model detection
- 🤝 Full OpenAI API compatibility
- ⚡ Blazing-fast startup (<0.5s)
- 📦 Ultra-compact binary (<20MB)

---

## 📦 Installation

<!--
### Script install
```bash
curl -sSL https://raw.githubusercontent.com/moyanj/tllama/main/install.sh | bash
```
-->
### Cargo install
```bash
cargo install tllama
```
<!--
### Pre-built binaries
Download from [Releases](https://github.com/moyanj/tllama/releases)
-->
---

## 🧪 Usage Guide
### Discover models
```bash
tllama discover [--all]
```

### Text generation
```bash
tllama infer <model_path> "<prompt>" \
  --n-len <tokens> \          # Output length (default: 128)
  --temperature <value> \     # Randomness (0-1)
  --top-k <value> \           # Top-k sampling
  --repeat-penalty <value>    # Repetition penalty
```

**Example:**
```bash
tllama infer ./llama3-8b.gguf "The future of AI is" \
  --temperature 0.7 \
  --n-len 256
```

### Interactive chat
```bash
tllama chat <model_path>
```

### Start API server
```bash
tllama serve \
  --host 0.0.0.0 \   # Bind address (default)
  --port 8080        # Port (default)
```

**Chat API Example:**
```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3-8b",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant"},
      {"role": "user", "content": "Explain Rust's memory safety"}
    ],
    "temperature": 0.7,
    "max_tokens": 200
  }'
```

---

## 📅 Development Roadmap
- [x] Core CLI implementation
- [x] GGUF quantized model support
- [ ] Model auto-download & caching
- [ ] Web UI integration
- [ ] Comprehensive test suite

---

## 🙌 Contributing
PRs welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines

---

## 🔐 License
MIT License

---

## ✨ Design Philosophy
> **Terminal-first**: Optimized for CLI workflows with 10x faster startup than Ollama
> **Minimal footprint**: Single binary under 5MB, zero external dependencies
> **Seamless integration**: Compatible with OpenAI SDKs and LangChain

---

## 📬 Contact
- GitHub: [moyanj/tllama](https://github.com/moyanj/tllama)
- Issues: [Report bugs](https://github.com/moyanj/tllama/issues)
- Feature requests: Open discussion issue

> ⭐ **Star us on GitHub to show your support!**