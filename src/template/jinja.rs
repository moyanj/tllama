use super::PromptData;
use minijinja::Environment;

const CHATML_TEMPLATE: &str = r#"{%- if Messages %}
{%- if System or Tools %}<|im_start|>system
{%- if System %}
{{ System }}
{%- endif %}
{%- if Tools %}

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{%- for Tool in Tools %}
{"type": "function", "function": {{ Tool.Function }}}
{%- endfor %}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>
{%- endif %}<|im_end|>
{% endif %}
{%- for message in Messages %}
{%- if message.Role == "user" %}<|im_start|>user
{{ message.Content }}<|im_end|>
{% elif message.Role == "assistant" %}<|im_start|>assistant
{% if message.Content %}{{ message.Content }}
{%- elif message.ToolCalls %}<tool_call>
{% for tool_call in message.ToolCalls %}{"name": "{{ tool_call.Function.Name }}", "arguments": {{ tool_call.Function.Arguments }}}
{% endfor %}</tool_call>
{%- endif %}{% if not loop.last %}<|im_end|>
{% endif %}
{%- elif message.Role == "tool" %}<|im_start|>user
<tool_response>
{{ message.Content }}
</tool_response><|im_end|>
{% endif %}
{%- if message.Role != "assistant" and loop.last %}<|im_start|>assistant
{% endif %}
{%- endfor %}
{%- else %}
{%- if System %}<|im_start|>system
{{ System }}<|im_end|>
{% endif %}{% if Prompt %}<|im_start|>user
{{ Prompt }}<|im_end|>
{% endif %}<|im_start|>assistant
{% endif %}{{ Response }}{% if Response %}<|im_end|>{{ endif }}
"#;

pub fn render_chatml_template(data: &PromptData) -> Result<String, Box<dyn std::error::Error>> {
    render_any_template(CHATML_TEMPLATE, data)
}

pub fn render_any_template(
    template: &str,
    data: &PromptData,
) -> Result<String, Box<dyn std::error::Error>> {
    let mut env = Environment::new();
    env.add_template("any", template)?;
    Ok(env.get_template("any")?.render(data)?)
}
