use super::TemplateData;
use minijinja::Environment;

pub fn render_any_template(
    template: &str,
    data: &TemplateData,
) -> Result<String, Box<dyn std::error::Error>> {
    let mut env = Environment::new();
    env.add_template("any", template)?;
    Ok(env.get_template("any")?.render(data)?)
}
