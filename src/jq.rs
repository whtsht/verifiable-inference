use serde_json::Error;
use std::io::Read;
use std::process::{Command, Stdio};

use serde::de::DeserializeOwned;

pub fn fetch_value<T: DeserializeOwned>(filename: &str, jq_command: &str) -> Result<T, Error> {
    let output = Command::new("jq")
        .arg(format!(".\"{}\"", jq_command))
        .arg(filename)
        .stdout(Stdio::piped())
        .spawn()
        .expect("Failed to execute jq");

    let mut output_data = String::new();
    let mut stdout = output.stdout.expect("Failed to open stdout");
    stdout
        .read_to_string(&mut output_data)
        .expect("Failed to read stdout");

    serde_json::from_str(&output_data)
}
