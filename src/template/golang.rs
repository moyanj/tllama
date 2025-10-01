use super::TemplateData;
use gotpl;

pub fn render_any_template(
    template: &str,
    data: &TemplateData,
) -> Result<String, Box<dyn std::error::Error>> {
    let mut data_clone = data.clone();
    if data_clone.response.is_none() {
        data_clone.response = Some("".to_string());
    }
    let rendered = gotpl::TemplateRenderer::new(template, &data_clone)
        .use_missing_key_zero(true)
        .render()?;
    Ok(rendered)
}
