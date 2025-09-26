use super::TemplateData;
use gtmpl_moyan;

use gtmpl_moyan::Value;
use serde::Serialize;
use std::collections::HashMap;

/// 将任何实现了 Serde Serialize 的类型转换为 gtmpl_value::Value
pub fn from_serde<T: Serialize>(value: T) -> Result<Value, serde_json::Error> {
    let json_value = serde_json::to_value(value)?;
    let gtmpl_value = json_value_to_gtmpl_value(json_value)?;
    Ok(gtmpl_value)
}

/// 将 serde_json::Value 转换为 gtmpl_value::Value
fn json_value_to_gtmpl_value(json_value: serde_json::Value) -> Result<Value, serde_json::Error> {
    match json_value {
        serde_json::Value::Null => Ok(Value::Nil),
        serde_json::Value::Bool(b) => Ok(Value::from(b)),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Ok(Value::from(i))
            } else if let Some(u) = n.as_u64() {
                Ok(Value::from(u))
            } else if let Some(f) = n.as_f64() {
                Ok(Value::from(f))
            } else {
                // 理论上不会发生，因为 serde_json::Number 总是上述类型之一
                Ok(Value::from(0))
            }
        }
        serde_json::Value::String(s) => Ok(Value::from(s)),
        serde_json::Value::Array(arr) => {
            let values: Result<Vec<Value>, _> =
                arr.into_iter().map(json_value_to_gtmpl_value).collect();
            Ok(Value::Array(values?))
        }
        serde_json::Value::Object(obj) => {
            let mut map = HashMap::new();
            for (key, value) in obj {
                map.insert(key, json_value_to_gtmpl_value(value)?);
            }
            Ok(Value::Map(map))
        }
    }
}

const CHATML_TEMPLATE: &str = r#"{{- if .Messages }}
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

pub fn render_chatml_template(data: &TemplateData) -> Result<String, Box<dyn std::error::Error>> {
    render_any_template(CHATML_TEMPLATE, data)
}

pub fn render_any_template(
    template: &str,
    data: &TemplateData,
) -> Result<String, Box<dyn std::error::Error>> {
    let mut data_clone = data.clone();
    if data_clone.response.is_none() {
        data_clone.response = Some("".to_string());
    }
    let rendered = gtmpl_moyan::template(template, from_serde(data_clone)?)?;
    Ok(rendered)
}
