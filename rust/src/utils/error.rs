use thiserror::Error;

#[derive(Error, Debug)]
#[allow(non_snake_case)]
pub enum ParameterError {
    #[error("Invalid number of dimensions (expected 1, 2, 3, got {K:?}")]
    InvalidNumDumensions {
        K: usize 
    },
}

#[derive(Error, Debug)]
pub enum RuntimeError {

    #[error("Failed to write to disk")]
    IOError,

    #[error("A NaN or Inf value was produced")]
    NanOrInf,

    #[error("Fourier aliasing occurred; p_mass is {p_mass:} and the threshold was set at {threshold:} with k2_cutoff = {k2_cutoff:}")]
    FourierAliasing {
        threshold: f32,
        k2_cutoff: f32,
        p_mass: f32
    },
}

#[macro_export]
macro_rules! Err {
    ($err:expr $(,)?) => {{
        let error = $err;
        Err(anyhow::anyhow!(error))
    }};
}
