use num::{ToPrimitive, FromPrimitive};
use serde_derive::Deserialize;
use cosmology::scale_factor::{
    CosmologicalParameters as InnerCosmoParams, ScaleFactor as InnerScaleFactorSolver
};
use num_traits::Float;

/// Only flat LCDM (w = 1) cosmologies are supported. 
/// An internal helper struct that is used to model the 
/// evolution of the cosmological scale factor a(t).
pub struct ScaleFactorSolver {

    /// Cosmological parameters retrieve from the toml
    pub cosmo_parameters: CosmologyParameters,

    // The next few are derived

    /// Omega_de now
    pub omega_de_now: f64,

    /// Initial time offshift from present
    pub t0: f64,
    
    /// On-the-fly solver
    pub solver: InnerScaleFactorSolver,
    
}

pub const DEFAULT_MAX_DLOGA: f64 = 1e-3;

#[derive(Deserialize, Debug, Clone, Copy)]
/// A helper struct. Deserialized from a simulation's toml file.
#[cfg(feature = "expanding")]
pub struct CosmologyParameters {

    /// Omega_matter now in units of critical density
    pub omega_matter_now: f64,

    /// Omega_radiation now in units of critical density
    pub omega_radiation_now: f64,

    /// Hubble constant now (little h)
    pub h: f64,

    /// Initial redshift at which simulation is initialized
    pub z0: f64,

    /// Solver parameter (maximum dloga change in one timestep). Default value of [`DEFAULT_MAX_DLOGA`].
    pub max_dloga: Option<f64>
}

impl CosmologyParameters {
    fn as_inner_params(&self) -> InnerCosmoParams {
        InnerCosmoParams {
            omega_m0: self.omega_matter_now,
            omega_de0: 1.0 - self.omega_matter_now - self.omega_radiation_now,
            omega_r0: self.omega_radiation_now,
            omega_k0: 0.0,
            w: 1.0,
            h: self.h,
        }
    }
}




impl ScaleFactorSolver {

    /// Given some parameters, constructs an internal helper struct that is used
    /// to model the evolution of the cosmological scale factor a(t)
    /// 
    /// Within [`CosmologyParameters`]:
    /// 
    /// `omega_matter_now: f64` is the current value of the total matter density
    /// (in units of the critical density). Sum of parameters must be less than
    /// or equal to unity, as only flat cosmologies are supported.
    /// 
    /// `omega_radiation_now: f64` is the current value of the total radiation density
    /// (in units of the critical density). Sum of parameters must be less than
    /// or equal to unity, as only flat cosmologies are supported.
    /// 
    /// `z0: f64` is the initial redshift at which the simulation
    /// is initialzed
    pub fn new(
        cosmo_parameters: CosmologyParameters
    ) -> ScaleFactorSolver {

        let omega_matter_now: f64 = cosmo_parameters.omega_matter_now;
        let omega_radiation_now: f64 = cosmo_parameters.omega_radiation_now;
        let z0: f64 = cosmo_parameters.z0;

        assert!(
            omega_matter_now + omega_radiation_now <= 1.0,
            "Only flat cosmologies are supported; pick an omega_matter_now + omega_radiation_now <= 1.0"
        );

        assert!(
            z0 >= 0.0,
            "Choose a positive initial redshift z0 >= 0.0"
        );

        assert!(
            omega_matter_now >= 0.0,
            "You've discovered matter with negative mass; pick an omega_matter_now >= 0.0"
        );

        assert!(
            omega_radiation_now >= 0.0,
            "You've discovered radiation with negative energy; pick an omega_radiation_now >= 0.0"
        );

        #[cfg(any(test, debug_assertions))]
        let t0 = 1.0;
        #[cfg(not(any(test, debug_assertions)))]
        let t0 = 1.0; //todo!();

        let solver = InnerScaleFactorSolver::new(
            cosmo_parameters.as_inner_params(),
            z0,
            cosmo_parameters.max_dloga.unwrap_or(DEFAULT_MAX_DLOGA),
            Some(t0),
        );

        ScaleFactorSolver {
            cosmo_parameters,
            omega_de_now: 1.0 - omega_matter_now - omega_radiation_now,
            t0,
            solver,
        }
    }

    /// Steps forward by `dt` and returns the value of the scale factor.
    #[cfg(not(feature = "fake_expanding_solver"))]
    fn step<T: Float + ToPrimitive + FromPrimitive>(&mut self, dt: T) -> T {

        // Step forward
        self.solver.step_forward(dt.to_f64().unwrap());

        // Return new a
        T::from_f64(self.solver.get_a()).unwrap()
    }

    /// Placeholder with scale_factor = 1 for all time.
    #[cfg(feature = "fake_expanding_solver")]
    fn step<T: Float + ToPrimitive + FromPrimitive>(&mut self, dt: T) -> T {
        T::one()
    }
}