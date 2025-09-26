use colored::*;
use rustyline::DefaultEditor;
use rustyline::error::ReadlineError;
use std::io::{Write, stdout};
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::Duration;

use crate::discover::Model;
use crate::{
    engine::{EngineConfig, InferenceEngine},
    template::*,
};

struct Spinner {
    handle: Option<thread::JoinHandle<()>>,
    stop_signal: Arc<AtomicBool>,
}

impl Spinner {
    /// 创建并启动一个新的 spinner 动画。
    /// message: 动画旁边显示的静态文本。
    fn new(message: String) -> Self {
        let stop_signal = Arc::new(AtomicBool::new(false));
        let signal_clone = stop_signal.clone();

        let handle = thread::spawn(move || {
            let spinner_chars = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'];
            let mut i = 0;
            // 隐藏光标以获得更好的视觉效果
            print!("\x1B[?25l");
            stdout().flush().unwrap();

            while !signal_clone.load(Ordering::Relaxed) {
                let frame = spinner_chars[i % spinner_chars.len()];
                // 使用 \r 将光标移回行首，实现原地更新
                print!("\r{} {}", message.dimmed(), frame);
                stdout().flush().unwrap();
                thread::sleep(Duration::from_millis(80));
                i += 1;
            }

            // 清理动画行并恢复光标
            // 使用空格覆盖整行内容
            print!("\r{}\r", " ".repeat(message.len() + 5));
            // 重新显示光标
            print!("\x1B[?25h");
            stdout().flush().unwrap();
        });

        Self {
            handle: Some(handle),
            stop_signal,
        }
    }

    /// 停止 spinner 动画并等待其清理完成。
    fn stop(mut self) {
        self.stop_signal.store(true, Ordering::Relaxed);
        if let Some(handle) = self.handle.take() {
            handle.join().unwrap();
        }
    }
}
// --- 模块结束 ---

struct ChatSession {
    engine: Box<dyn InferenceEngine>,
    data: Vec<Message>,
    system_prompt: String,
}

impl ChatSession {
    fn new(engine: Box<dyn InferenceEngine>) -> Self {
        Self {
            engine,
            data: vec![],
            system_prompt: "You are a helpful, respectful and honest AI assistant.".to_string(),
        }
    }

    fn print_welcome_message(&self) {
        println!("{}", "========================================".cyan());
        println!("{}", " Welcome to Rllama!".cyan().bold());
        println!("{}", "========================================".cyan());
        println!("Type your message and press Enter to chat with the AI.");
        println!("Type `.help` for more commands.");
        println!("Type `.exit` or press Ctrl+C to quit.");
        println!();
    }

    fn handle_command(&mut self, command: &str) -> Result<bool, Box<dyn std::error::Error>> {
        let parts: Vec<&str> = command.trim().splitn(2, ' ').collect();
        let cmd = parts[0];

        match cmd {
            ".exit" | ".quit" | ".q" | ".bye" => {
                println!("{}", "Goodbye!".yellow());
                return Ok(false);
            }
            ".help" => {
                println!("{}", "Available Commands:".green().bold());
                println!("  {:<15} {}", ".help", "Show this help message.");
                println!(
                    "  {:<15} {}",
                    ".system [prompt]", "View or set the system prompt."
                );
                println!("  {:<15} {}", ".clear", "Clear the conversation history.");
                println!("  {:<15} {}", ".history", "Show the conversation history.");
                println!("  {:<15} {}", ".exit", "Exit the chat session.");
            }
            ".system" => {
                if let Some(new_prompt) = parts.get(1) {
                    self.system_prompt = new_prompt.to_string();
                    println!(
                        "{} {}",
                        "System prompt updated:".green(),
                        self.system_prompt
                    );
                } else {
                    println!(
                        "{} {}",
                        "Current system prompt:".green(),
                        self.system_prompt
                    );
                }
            }
            ".clear" => {
                self.data.clear();
                println!("{}", "Conversation history cleared.".green());
            }
            ".history" => {
                println!("{}", "Conversation History:".green().bold());
                if self.data.is_empty() {
                    println!("  (No history yet)");
                } else {
                    for msg in &self.data {
                        let prefix = if msg.role == "user" {
                            "You".blue()
                        } else {
                            "AI".cyan()
                        };
                        println!("{}: {}", prefix, msg.content.as_deref().unwrap_or(""));
                    }
                }
            }
            _ => {
                println!("{}'{}'", "Unknown command: ".red(), command);
            }
        }
        Ok(true)
    }

    fn start(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let mut rl = DefaultEditor::new()?;
        ctrlc::set_handler(move || {
            println!("\n{}", "Received Ctrl-C. Exiting...".yellow());
            // 确保退出时恢复光标
            print!("\x1B[?25h");
            stdout().flush().unwrap();
            std::process::exit(0);
        })?;

        self.print_welcome_message();

        loop {
            let readline = rl.readline(&">>> ".green().to_string());
            match readline {
                Ok(line) => {
                    let input = line.trim();
                    if input.is_empty() {
                        continue;
                    }

                    rl.add_history_entry(input)?;

                    if input.starts_with('.') {
                        if !self.handle_command(input)? {
                            break;
                        }
                    } else {
                        self.chat(input)?;
                    }
                }
                Err(ReadlineError::Interrupted) => {
                    println!("{}", "Received Ctrl-C. Exiting...".yellow());
                    break;
                }
                Err(ReadlineError::Eof) => {
                    println!("{}", "Received Ctrl-D. Exiting...".yellow());
                    break;
                }
                Err(err) => {
                    println!("Error: {:?}", err);
                    break;
                }
            }
        }
        Ok(())
    }

    fn chat(&mut self, user_input: &str) -> Result<(), Box<dyn std::error::Error>> {
        self.data.push(Message {
            role: "user".to_string(),
            content: Some(user_input.to_string()),
            tool_calls: None,
            name: None,
        });

        let prompt_data = TemplateData::new()
            .with_system(Some(self.system_prompt.clone()))
            .with_messages(Some(self.data.clone()));

        let prompt = render_chatml_template(&prompt_data)?;

        // --- 修改: 在AI响应前启动动画 ---
        // 将 Spinner 包装在 Option 中，以便在闭包内安全地 `.take()` 和消耗它。
        let spinner = Arc::new(Mutex::new(Some(Spinner::new("".to_string()))));
        let spinner_clone = spinner.clone();
        let mut first_token = true;

        let result = self.engine.infer(
            &prompt,
            None,
            crate::def_callback!(|token| {
                if first_token {
                    // 在接收到第一个 token 时，停止并移除 spinner。
                    let mut spinner_guard = spinner_clone.lock().unwrap();
                    if let Some(s) = spinner_guard.take() {
                        s.stop();
                    }
                    stdout().flush().unwrap();
                    first_token = false;
                }
                // 流式打印 AI 的回复
                print!("{}", token);
                stdout().flush().unwrap();
            }),
        );

        // 如果流式传输结束但没有收到任何 token（例如出错或空回复），
        // 确保 spinner 仍然被停止。
        let mut spinner_guard = spinner.lock().unwrap();
        if let Some(s) = spinner_guard.take() {
            s.stop();
        }

        println!(); // 在 AI 回复结束后换行

        self.data.push(Message {
            role: "assistant".to_string(),
            content: Some(result?),
            tool_calls: None,
            name: None,
        });
        Ok(())
    }
}

pub fn chat_session(args: crate::cli::ChatArgs) -> Result<(), Box<dyn std::error::Error>> {
    llama_cpp_2::send_logs_to_tracing(llama_cpp_2::LogOptions::default().with_logs_enabled(false));
    let model_path;
    if args.model.starts_with('.') || args.model.starts_with('/') {
        model_path = Model::from_path(&args.model)
    } else {
        model_path = crate::discover::MODEL_DISCOVERER
            .lock()
            .unwrap()
            .find_model(&args.model)?;
    }

    let engine_config = EngineConfig {
        n_ctx: 2048,
        n_len: None,
        temperature: 0.8,
        top_k: 40,
        top_p: 0.9,
        repeat_penalty: 1.1,
    };

    // --- 修改: 在加载模型时使用动画 ---
    // 1. 启动 spinner
    let spinner = Spinner::new("Loading model...".to_string());

    // 2. 执行耗时操作
    let engine_result = crate::engine::llama_cpp::LlamaEngine::new(&engine_config, &model_path);

    // 3. 停止 spinner
    spinner.stop();

    // 4. 根据结果打印最终信息
    let engine = match engine_result {
        Ok(engine) => {
            println!("{} {}", "Model loaded successfully.".dimmed(), "✔".green());
            Box::new(engine)
        }
        Err(e) => {
            eprintln!("\n{} {}", "Failed to load model.".red().bold(), "✖".red());
            return Err(e.into());
        }
    };
    // --- 修改结束 ---

    let mut session = ChatSession::new(engine);
    session.start()
}
