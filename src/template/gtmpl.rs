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
