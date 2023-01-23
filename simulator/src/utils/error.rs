use msm_common::CommonError;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum RuntimeError {
    #[error("Failed to write to disk")]
    IOError,

    #[error("A NaN or Inf value was produced")]
    NanOrInf,

    #[error("Fourier aliasing occurred; p_mass is {p_mass:} and the threshold was set at {threshold:} with k2_cutoff = {k2_cutoff:}")]
    FourierAliasing {
        threshold: f64,
        k2_cutoff: f64,
        p_mass: f64,
    },

    #[error("Error in common: {err}")]
    TomlReadError {
        #[from]
        err: CommonError,
    },

    #[error("Failed to retrieve keypair {e}")]
    KeypairError { e: Box<dyn std::error::Error> },
}

#[macro_export]
macro_rules! Err {
    ($err:expr $(,)?) => {{
        let error = $err;
        Err(anyhow::anyhow!(error))
    }};
}
