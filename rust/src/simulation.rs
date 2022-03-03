use ndarray::Array;
use num_traits::Float;

/// In the original python implementation, this was a `sim` object.
/// Here, we rename it SpatialGrid, since it is just a spatial grid
/// (and its fourier transform) with several fields
pub struct Simulation<T: Float, U> {//, const K: usize> {

    // Spatial Fields
    /// The wavefunction
    pub ψ: Array<T, U>,

    /// Fourier
    pub k_grid: Array<T, U>,

}