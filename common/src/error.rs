use thiserror::Error;

#[derive(Debug, Error)]
pub enum CommonError {
    #[error("Unable to load toml: {path}")]
    TomlReadError { path: String },

    #[error("Unable to parse toml: {msg}")]
    TomlParseError { msg: String },
}
