use serde_derive::Deserialize;
use cosmology;

/// Only flat LCDM (w = 1) cosmologies are supported. 
/// An internal helper struct that is used to model the 
/// evolution of the cosmological scale factor a(t).
#[derive(Deserialize)]
pub struct CosmologyParameters {

    /// Omega_matter now
    omega_matter_now: f64,

    /// Omega_de now
    omega_de_now: f64,
    
    /// Initial redshift at which simulation is initialized
    z0: f64
}

impl CosmologyParameters {

    /// Given some parameters, constructs an internal helper struct that is used
    /// to model the evolution of the cosmological scale factor a(t)
    /// 
    /// 
    /// `omega_matter_now: f64` is the current value of the total matter density
    /// (in units of the critical density). Must be less than or equal to unity,
    /// as only flat cosmologies are supported.
    /// 
    /// `z0: f64` is the initial redshift at which the simulation
    /// is initialzed
    pub fn new(omega_matter_now: f64, z0: f64) -> CosmologyParameters {

        assert!(
            omega_matter_now <= 1.0,
            "Only flat cosmologies are supported; pick an omega_matter_now <= 1.0"
        );

        assert!(
            z0 >= 0.0,
            "Choose a positive initial redshift z0 >= 0.0"
        );

        assert!(
            omega_matter_now >= 0.0,
            "You've discovered matter with negative mass; pick an omega_matter_now >= 0.0"
        );

        CosmologyParameters {
            omega_matter_now,
            omega_de_now: 1.0 - omega_matter_now,
            z0,
        }
    }
}