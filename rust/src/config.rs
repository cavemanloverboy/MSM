use num_traits::Float;

/// This `Configuration` struct stores simulations parameters
pub struct Configuration<T: Float> {
    // Grid Parameters
    /// Number of pixels per axis
    pub n_grid: u32,
    /// Physical length of each axis
    pub axis_length: T,
    /// Spatial cell size
    pub dx: T,
    /// k-space cell size
    pub dk: T,

    // Temporal Parameters
    /// Total simulation time
    pub total_sim_time: T,
    /// Number of data dumps
    pub num_data_dumps: u32,

    // Physical Parameters
    pub total_mass: T,
    // Particle mass
    pub particle_mass: T,

    // Metadata
    /// Simulation name
    pub sim_name: &'static str,
    /// Precision
    pub float_precision: &'static str,
}
