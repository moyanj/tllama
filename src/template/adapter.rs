use crate::{
    discover::{Model, ModelType},
    template::{RenderFunc, TemplateData},
};

#[cfg(feature = "engine-hf")]
use super::jinja::render_any_template as hf_render;

#[cfg(feature = "tpl-gtmpl")]
use super::gtmpl::render_any_template as gtmpl_render;

#[cfg(feature = "tpl-gotpl")]
use super::golang::render_any_template as gotpl_render;

fn select_engine(model: &Model) -> RenderFunc {
    match model.format {
        #[cfg(feature = "engine-hf")]
        ModelType::Transformers => {
            return hf_render;
        }

        ModelType::Gguf => {
            #[cfg(feature = "tpl-gotpl")]
            return gotpl_render;

            #[cfg(feature = "tpl-gtmpl")]
            return gtmpl_render;

            #[cfg(not(any(feature = "tpl-gtmpl", feature = "tpl-gotpl")))]
            compile_error!(
                "GGUF models require either `tpl-gtmpl` or `tpl-gotpl` feature to be enabled."
            );
        }

        #[cfg(not(feature = "engine-hf"))]
        ModelType::Transformers => {
            compile_error!("Transformers models require the `engine-hf` feature to be enabled.")
        }
    }
}

#[cfg(any(feature = "tpl-gtmpl", feature = "tpl-gotpl"))]
pub const GO_DEFAULT_TEMPLATE: &str = r#"{{- if .Messages }}
{{- if or .System .Tools }}<|im_start|>system
{{- if .System }}
{{ .System }}
{{- end }}
{{- if .Tools }}

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{{- range .Tools }}
{"type": "function", "function": {{ .Function }}}
{{- end }}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>
{{- end }}<|im_end|>
{{ end }}
{{- range $i, $_ := .Messages }}
{{- $last := eq (len (slice $.Messages $i)) 1 -}}
{{- if eq .Role "user" }}<|im_start|>user
{{ .Content }}<|im_end|>
{{ else if eq .Role "assistant" }}<|im_start|>assistant
{{ if .Content }}{{ .Content }}
{{- else if .ToolCalls }}<tool_call>
{{ range .ToolCalls }}{"name": "{{ .Function.Name }}", "arguments": {{ .Function.Arguments }}}
{{ end }}</tool_call>
{{- end }}{{ if not $last }}<|im_end|>
{{ end }}
{{- else if eq .Role "tool" }}<|im_start|>user
<tool_response>
{{ .Content }}
</tool_response><|im_end|>
{{ end }}
{{- if and (ne .Role "assistant") $last }}<|im_start|>assistant
{{ end }}
{{- end }}
{{- else }}
{{- if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ if .Prompt }}<|im_start|>user
{{ .Prompt }}<|im_end|>
{{ end }}<|im_start|>assistant
{{ end }}{{ .Response }}{{ if .Response }}<|im_end|>{{ end }}
"#;

#[cfg(feature = "tpl-minijinja")]
pub const JINJA_DEFAULT_TEMPLATE: &str = r#"{%- if Messages %}
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

pub fn get_default_template() -> String {
    #[cfg(any(feature = "tpl-gtmpl", feature = "tpl-gotpl"))]
    return GO_DEFAULT_TEMPLATE.to_string();
    #[cfg(feature = "tpl-minijinja")]
    #[allow(unreachable_code)]
    return JINJA_DEFAULT_TEMPLATE.to_string();
}

pub fn render_template(
    model: &Model,
    template: &str,
    data: &TemplateData,
) -> Result<String, Box<dyn std::error::Error>> {
    println!("{}", template);
    let render_func = select_engine(&model);
    render_func(template, data)
}
