use arrayfire::{
    Array, ComplexFloating, HasAfEnum, FloatingPoint, ConstGenerator, Dim4,
    mul, real, conjg, constant
};
use crate::{
    constants::POIS_CONST,
    utils::{
        fft::{forward, inverse, forward_inplace, inverse_inplace, inv_spec_grid},
        complex::complex_constant,
        io,
        error::MSMError,
    }
};
use conv::*;
use std::fmt::Display;
use num::{Complex, Float, FromPrimitive};

/// This struct holds the grids which store the wavefunction and its Fourier transform
pub struct SimulationGrid<T, const K: usize, const S: usize>
where
    T: Float + FloatingPoint + ConstGenerator<OutType=T>,
    Complex<T>: HasAfEnum + FloatingPoint + HasAfEnum<AbsOutType = T>,
{

    // Spatial Fields
    /// The array which stores the wavefunction
    pub ψ: Array<Complex<T>>,

    /// Fourier space
    pub ψk: Array<Complex<T>>,
}



/// This `Parameters` struct stores simulations parameters
pub struct SimulationParameters<U: Float + FloatingPoint> {

    // Grid Parameters
    /// Number of pixels per axis
    pub n_grid: u32,
    /// Physical length of each axis
    pub axis_length: U,
    /// Spatial cell size
    pub dx: U,
    /// k-space cell size
    pub dk: U,

    // Temporal Parameters
    /// Current simulation time
    pub time: U,
    /// Total simulation time
    pub total_sim_time: U,
    /// Number of data dumps
    pub num_data_dumps: u32,

    // Physical Parameters
    /// Total Mass
    pub total_mass: U,
    /// Particle mass
    pub particle_mass: U,

    // Metadata
    /// Simulation name
    pub sim_name: &'static str,
}

/// In the original python implementation, this was a `sim` or `SimObject` object.
/// This stores a `SimulationGrid` which has the wavefunction and its fourier transform.
/// It also holds the `SimulationParameters` which holds the simulation parameters.
pub struct SimulationObject<T, const K: usize, const S: usize>
where
    T: Float + FloatingPoint + ConstGenerator<OutType=T>,
    Complex<T>: HasAfEnum + ComplexFloating + FloatingPoint + HasAfEnum<AbsOutType = T>,
{

    /// This has the wavefunction and its Fourier transform
    pub grid: SimulationGrid<T, K, S>,

    /// This has the simulation parameters
    pub parameters: SimulationParameters<T>,

}

impl<T, const K: usize, const S: usize> SimulationGrid<T, K, S>
where
    T: Float + FloatingPoint + ConstGenerator<OutType=T>,
    Complex<T>: HasAfEnum + ComplexFloating + FloatingPoint + HasAfEnum<ComplexOutType = Complex<T>> + HasAfEnum<AbsOutType = T>,
 {

    pub fn new(
        ψ: Array<Complex<T>>, 
    ) -> Self
    where
        T: Float,
        Complex<T>: HasAfEnum + ComplexFloating + FloatingPoint,
        <Complex<T> as arrayfire::HasAfEnum>::ComplexOutType: HasAfEnum
    {
        let ψk = forward::<T, K, S>(&ψ).expect("failed forward fft");
        SimulationGrid {
            ψ,
            ψk,
        }
    }

}

impl<U> SimulationParameters<U>
where
    U: FromPrimitive + Float + FloatingPoint + Display
{

    pub fn new(
        n_grid: u32,
        axis_length: U,
        time: U,
        total_sim_time: U,
        num_data_dumps: u32,
        total_mass: U,
        particle_mass: U,
        sim_name: &'static str,

    ) -> Self
    {

        // Overconstrained parameters 
        let dx = axis_length / U::from_u32(n_grid).unwrap();
        let dk = U::from_f64(2.0).unwrap() * U::from_f64(std::f64::consts::PI).unwrap() / dx;

        SimulationParameters {
            n_grid,
            axis_length,
            dx,
            dk,
            time,
            total_sim_time,
            num_data_dumps,
            total_mass,
            particle_mass,
            sim_name, 
        }
    }
}
impl<U> Display for SimulationParameters<U>
where
    U: ValueFrom<u32> + ValueFrom<f64> + Float + FloatingPoint + Display
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}\n","-".repeat(40))?;
        write!(f, "n_grid         = {}\n", self.n_grid)?;
        write!(f, "axis_length    = {}\n", self.axis_length)?;
        write!(f, "dx             = {}\n", self.dx)?;
        write!(f, "dk             = {}\n", self.dk)?;
        write!(f, "current_time   = {}\n", self.time)?;
        write!(f, "total_sim_time = {}\n", self.total_sim_time)?;
        write!(f, "num_data_dumps = {}\n", self.num_data_dumps)?;
        write!(f, "total_mass     = {}\n", self.total_mass)?;
        write!(f, "particle_mass  = {}\n", self.particle_mass)?;
        write!(f, "sim_name       = {}\n", self.sim_name,)?;
        write!(f, "{}\n","-".repeat(40))?;
        Ok(())
    }
}



impl<T, const K: usize, const S: usize> SimulationObject<T, K, S>
where
    T: Float + FloatingPoint + Display + FromPrimitive + ConstGenerator<OutType=T>,
    Complex<T>: HasAfEnum + FloatingPoint + ComplexFloating + HasAfEnum<ComplexOutType = Complex<T>> + HasAfEnum<AbsOutType = T>,
{

    pub fn new(
        ψ: Array<Complex<T>>,
        n_grid: u32,
        axis_length: T,
        time: T,
        total_sim_time: T,
        num_data_dumps: u32,
        total_mass: T,
        particle_mass: T,
        sim_name: &'static str,
    ) -> Self {
        
        // Construct components
        let grid = SimulationGrid::<T, K, S>::new(ψ);
        let parameters = SimulationParameters::<T>::new(
            n_grid,
            axis_length,
            time,
            total_sim_time,
            num_data_dumps,
            total_mass,
            particle_mass,
            sim_name,
        );

        SimulationObject {
            grid,
            parameters,
        }
    }

    /// This function updates the `SimulationGrid` stored in the `SimulationObject`.
    pub fn update(&mut self) {


    }

    /// This function computes the shape of the grid
    pub fn get_shape(&self) -> Result<(u64, u64, u64, u64), MSMError> {
        match K {
            1 => Ok((S as u64, 1, 1, 1)),
            2 => Ok((S as u64, S as u64, 1, 1)),
            3 => Ok((S as u64, S as u64, S as u64, 1)),
            _ => Err(MSMError::InvalidNumDumensions(K))
        }
    }

    /// This function computes the space density 
    pub fn get_density(&self) -> Array<T> {
        let shape = self.get_shape().unwrap();
        mul(
            &Array::new(
                    &[self.parameters.total_mass],
                    Dim4::new(&[1, 1, 1, 1])
            ),
            &real(
                &mul(
                    &self.grid.ψ,
                    &conjg(&self.grid.ψ),
                    false
                )
            ),
            true
        )
    }

    /// This function calculates the potential for the stream
    pub fn calculate_potential(&self) -> Array<T> {

        // Compute space density and perform inplace fft
        let mut ρ: Array<Complex<T>> = self.get_density().cast();
        forward_inplace::<T, K, S>(&mut ρ);

        // Compute potential in k-space and perform inplace inverse fft
        let mut φ: Array<Complex<T>> = mul(
            &mul(
                &Array::new(
                    &[Complex::<T>::new(
                        -T::from_f64(POIS_CONST).unwrap(),
                        T::zero()
                    )],
                    Dim4::new(&[1,1,1,1])
                ),
                &ρ,
                true
            ),
            &self.get_inv_spec_grid().cast(),
            false
        );
        inverse_inplace::<T, K, S>(&mut φ);
        real(&φ)
    }

    pub fn get_inv_spec_grid(&self) -> Array<T> {
        inv_spec_grid::<T, K, S>(self.parameters.dx, self.get_shape().unwrap())
    }
}


#[test]
fn test_new_grid() {

    use arrayfire::Dim4;
    //use arrayfire::af_print;

    // Grid parameters
    const K: usize = 1;
    const S: usize = 32;

    // Random wavefunction
    let values = [Complex::<f32>::new(1.0, 2.0); S];
    let dims = Dim4::new(&[S as u64, 1, 1, 1]);
    let ψ: Array<Complex<f32>> = Array::new(&values, dims);

    // Initialize grid
    let _grid: SimulationGrid<f32, K, S> = SimulationGrid::<f32, K, S>::new(ψ);
    //af_print!("ψ", grid.ψ);
    //af_print!("ψk", grid.ψk);
}


#[test]
fn test_new_sim_parameters() {

    type T = f64;

    let n_grid: u32 = 16;
    let axis_length: T = 1.0; 
    let time: T = 0.0;
    let total_sim_time: T = 1.0;
    let num_data_dumps: u32 = 100;
    let total_mass: T = 1.0;
    let particle_mass: T = 1e-6;
    let sim_name: &'static str = "my-sim";

    let params = SimulationParameters::new(
        n_grid,
        axis_length,
        time,
        total_sim_time,
        num_data_dumps,
        total_mass,
        particle_mass,
        sim_name,
    );
    println!("{}", params);
}


