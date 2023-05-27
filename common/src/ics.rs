use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(tag = "type")]
pub enum InitialConditions {
    /// Loads user specified initial conditions
    UserSpecified { path: String },

    /// A real (phases = 0) gaussian in real space
    ColdGauss { mean: Vec<f64>, std: Vec<f64> },

    /// A gaussian in fourier space with uniform random phases determined by `phase_seed`.
    ColdGaussKSpace {
        mean: Vec<f64>,
        std: Vec<f64>,
        phase_seed: Option<u64>,
    },

    /// A spherical tophat in real space
    SphericalTophat { radius: f64, delta: f64, slope: f64 },
}

#[derive(Serialize, Deserialize, Debug)]
pub struct SamplingParameters {
    /// The seed used by the rng during sampling
    pub seed: u64,
    /// The quantum sampling scheme to use, e.g Wigner
    pub scheme: SamplingScheme,
}


#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq)]
pub enum SamplingScheme {
    Poisson,
    Wigner,
    Husimi,
}
