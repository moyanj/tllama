use crate::{discover::Model, env::TLLAMA_RPC_HOST};
use lazy_static::lazy_static;
use reqwest::StatusCode;

lazy_static! {
    pub static ref RPC_CLIENT: RPCClient = RPCClient::new();
}

pub struct RPCClient {
    client: reqwest::blocking::Client,
    base_url: String,
}

impl RPCClient {
    pub fn new() -> Self {
        Self {
            client: reqwest::blocking::Client::new(),
            base_url: TLLAMA_RPC_HOST.to_string(),
        }
    }

    pub fn discover(&self) -> Result<bool, reqwest::Error> {
        let url = format!("{}/discover", self.base_url);
        let status = self.client.get(url).send()?.status();
        match status {
            StatusCode::OK => Ok(true),
            _ => Ok(false),
        }
    }

    pub fn list(&self) -> Result<Vec<Model>, reqwest::Error> {
        let url = format!("{}/list", self.base_url);
        let response = self.client.get(url).send()?;
        let status = response.status();
        match status {
            StatusCode::OK => Ok(serde_json::from_str(response.text().unwrap().as_str()).unwrap()),
            _ => Err(response.error_for_status().unwrap_err()),
        }
    }
}
