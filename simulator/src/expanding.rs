use cosmology::scale_factor::{
    CosmologicalParameters as InnerCosmoParams, ScaleFactor as InnerScaleFactorSolver,
};
use msm_common::CosmologyParameters;
use num::{FromPrimitive, ToPrimitive};
use num_traits::Float;

/// Only flat LCDM (w = 1) cosmologies are supported.
/// An internal helper struct that is used to model the
/// evolution of the cosmological scale factor a(t).
#[derive(Clone)]
pub struct ScaleFactorSolver {
    /// Cosmological parameters retrieve from the toml
    pub cosmo_parameters: CosmologyParameters,

    // The next few are derived
    /// Omega_de now
    pub omega_de_now: f64,

    /// Initial time offshift from present
    pub t0: f64,

    /// On-the-fly solver
    solver: InnerScaleFactorSolver,
}

pub const DEFAULT_MAX_DLOGA: f64 = 1e-3;

fn from_params(params: CosmologyParameters) -> InnerCosmoParams {
    InnerCosmoParams {
        omega_m0: params.omega_matter_now,
        omega_de0: 1.0 - params.omega_matter_now - params.omega_radiation_now,
        omega_r0: params.omega_radiation_now,
        omega_k0: 0.0,
        w: 1.0,
        h: params.h,
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
    pub fn new(cosmo_parameters: CosmologyParameters) -> ScaleFactorSolver {
        let omega_matter_now: f64 = cosmo_parameters.omega_matter_now;
        let omega_radiation_now: f64 = cosmo_parameters.omega_radiation_now;
        let z0: f64 = cosmo_parameters.z0;

        assert!(
            omega_matter_now + omega_radiation_now <= 1.0,
            "Only flat cosmologies are supported; pick an omega_matter_now + omega_radiation_now <= 1.0"
        );

        assert!(z0 >= 0.0, "Choose a positive initial redshift z0 >= 0.0");

        assert!(
            omega_matter_now >= 0.0,
            "You've discovered matter with negative mass; pick an omega_matter_now >= 0.0"
        );

        assert!(
            omega_radiation_now >= 0.0,
            "You've discovered radiation with negative energy; pick an omega_radiation_now >= 0.0"
        );

        // a(t_0) = 1 / ( 1 + z )
        let t0 = 0.0;

        let solver = InnerScaleFactorSolver::new(
            from_params(cosmo_parameters),
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
}

impl ScaleFactorSolver {
    /// Steps forward by `dt` and returns the value of the scale factor.
    pub(crate) fn step<T: Float + ToPrimitive + FromPrimitive>(&mut self, dt: T) -> T {
        // Step forward
        self.solver.step_forward(dt.to_f64().unwrap());

        // Return new a
        T::from_f64(self.solver.get_a()).unwrap()
    }

    pub(crate) fn get_a(&self) -> f64 {
        self.solver.get_a()
    }

    pub(crate) fn get_dadt(&self) -> f64 {
        self.solver.get_dadt()
    }

    pub(crate) fn get_time(&self) -> f64 {
        self.solver.get_time()
    }
}
