use arrayfire::{
    Array, ComplexFloating, HasAfEnum, FloatingPoint, ConstGenerator, Dim4, Fromf64,
    mul, real, conjg, constant, exp, max_all, div, replace_scalar, isnan, min_all, bitnot, sum_all
};
use crate::{
    constants::{POIS_CONST, HBAR},
    utils::{
        fft::{forward, inverse, forward_inplace, inverse_inplace, spec_grid, get_kgrid},
        complex::complex_constant,
        io,
        error::MSMError,
    }
};
use conv::*;
use std::fmt::Display;
use num::{Complex, Float, FromPrimitive};
use std::time::Instant;

/// This struct holds the grids which store the wavefunction and its Fourier transform
pub struct SimulationGrid<T, const K: usize, const S: usize>
where
    T: Float + FloatingPoint + ConstGenerator<OutType=T> + HasAfEnum<InType = T> + HasAfEnum<BaseType = T> + Fromf64,
    Complex<T>: HasAfEnum + FloatingPoint + HasAfEnum<AbsOutType = T>,
{

    // Spatial Fields
    /// The array which stores the wavefunction
    pub ψ: Array<Complex<T>>,

    /// Fourier space
    pub ψk: Array<Complex<T>>,
}



/// This `Parameters` struct stores simulations parameters
pub struct SimulationParameters<U: Float + FloatingPoint, const S: usize> {

    // Grid Parameters
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
    /// Total number of data dumps
    pub num_data_dumps: u32,
    /// Current number of data dumps
    pub current_dumps: u32,
    /// Timestep Criterion
    pub cfl: U,

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
    T: Float + FloatingPoint + ConstGenerator<OutType=T> + HasAfEnum<InType = T> + HasAfEnum<BaseType = T> + Fromf64,
    Complex<T>: HasAfEnum + ComplexFloating + FloatingPoint + HasAfEnum<AbsOutType = T>,
{

    /// This has the wavefunction and its Fourier transform
    pub grid: SimulationGrid<T, K, S>,

    /// This has the simulation parameters
    pub parameters: SimulationParameters<T, S>,

}

impl<T, const K: usize, const S: usize> SimulationGrid<T, K, S>
where
    T: Float + FloatingPoint + ConstGenerator<OutType=T> + HasAfEnum<InType = T> + HasAfEnum<BaseType = T> + Fromf64,
    Complex<T>: HasAfEnum + ComplexFloating + FloatingPoint + HasAfEnum<ComplexOutType = Complex<T>> + HasAfEnum<UnaryOutType = Complex<T>> + HasAfEnum<AbsOutType = T>,
 {

    pub fn new(
        ψ: Array<Complex<T>>, 
    ) -> Self
    {
        let ψk = forward::<T, K, S>(&ψ).expect("failed forward fft");
        SimulationGrid {
            ψ,
            ψk,
        }
    }

}

impl<U, const S: usize> SimulationParameters<U, S>
where
    U: FromPrimitive + Float + FloatingPoint + Display
{

    pub fn new(
        axis_length: U,
        time: U,
        total_sim_time: U,
        cfl: U,
        num_data_dumps: u32,
        total_mass: U,
        particle_mass: U,
        sim_name: &'static str,

    ) -> Self
    {

        // Overconstrained parameters 
        let dx = axis_length / U::from_usize(S).unwrap();
        let dk = U::from_f64(2.0).unwrap() * U::from_f64(std::f64::consts::PI).unwrap() / axis_length;
        let current_dumps = 0;

        SimulationParameters {
            axis_length,
            dx,
            dk,
            time,
            total_sim_time,
            cfl,
            num_data_dumps,
            current_dumps,
            total_mass,
            particle_mass,
            sim_name, 
        }
    }
}
impl<U, const S: usize> Display for SimulationParameters<U, S>
where
    U: ValueFrom<u32> + ValueFrom<f64> + Float + FloatingPoint + Display
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}\n","-".repeat(40))?;
        write!(f, "axis_length    = {}\n", self.axis_length)?;
        write!(f, "dx             = {}\n", self.dx)?;
        write!(f, "dk             = {}\n", self.dk)?;
        write!(f, "current_time   = {}\n", self.time)?;
        write!(f, "total_sim_time = {}\n", self.total_sim_time)?;
        write!(f, "cfl            = {}\n", self.cfl)?;
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
    T: Float + FloatingPoint + Display + FromPrimitive + ConstGenerator<OutType=T> + HasAfEnum<InType = T> + HasAfEnum<BaseType = T> + Fromf64 + ndarray_npy::WritableElement,
    Complex<T>: HasAfEnum + FloatingPoint + ComplexFloating + HasAfEnum<ComplexOutType = Complex<T>> + HasAfEnum<UnaryOutType = Complex<T>> + HasAfEnum<AbsOutType = T>,
{

    pub fn new(
        ψ: Array<Complex<T>>,
        axis_length: T,
        time: T,
        total_sim_time: T,
        cfl: T,
        num_data_dumps: u32,
        total_mass: T,
        particle_mass: T,
        sim_name: &'static str,
    ) -> Self {
        
        // Construct components
        let grid = SimulationGrid::<T, K, S>::new(ψ);
        let parameters = SimulationParameters::<T, S>::new(
            axis_length,
            time,
            total_sim_time,
            cfl,
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
        let now = Instant::now();

        // Used throughout. TODO: perhaps make a field of SimulationGrid
        let shape = self.get_shape().unwrap();

        // Calculate potential
        let φ: Array<T> = self.calculate_potential();
        let v: Array<T> = mul(&φ, &constant!(self.parameters.particle_mass; shape.0, shape.1, shape.2, shape.3), false);


        // Compute timestep
        let (dump, dt) = self.get_timestep(&v);

        // Update spatial wavefunction half-step
        // TODO: compare dt to 
        let r_evolution: Array<Complex<T>> = exp(
            &mul(
                &complex_constant(Complex::<T>::new(T::zero(), -(dt / T::from_f64(2.0).unwrap()) / T::from_f64(2.0 * HBAR).unwrap()), shape),
                &v.cast(),
                true
            )
        );
        self.grid.ψ = mul(&self.grid.ψ, &r_evolution, false);

        // Update momentum full-step
        self.grid.ψk = forward::<T, K, S>(&self.grid.ψ).unwrap();
        let k_evolution: Array<Complex<T>> = exp(
            &mul(
                &complex_constant(Complex::<T>::new(T::zero(), -(dt / (T::from_f64(2.0 / HBAR).unwrap() * self.parameters.particle_mass)) / T::from_f64(2.0 * HBAR).unwrap()), shape),
                &self.get_spec_grid().cast(),
                true
            )
        );
        self.grid.ψk = mul(&self.grid.ψk, &k_evolution, false);
        self.grid.ψ = inverse::<T, K, S>(&self.grid.ψk).unwrap();

        // Update position half-step
        self.grid.ψ = mul(&self.grid.ψ, &r_evolution, false);
        self.parameters.time = self.parameters.time + dt;

        //let estimate = now.elapsed().as_millis() * T::to_u128(&((self.parameters.total_sim_time - self.parameters.time)/dt)).unwrap();
        //println!("update took {} millis, current sim time is {}, dt is {}. ETA {:?} ", now.elapsed().as_millis(), self.parameters.time, dt, std::time::Duration::from_millis(estimate as u64));
        println!("update took {} millis, current sim time is {}, dt is {}", now.elapsed().as_millis(), self.parameters.time, dt);

        if dump {
            self.dump();
            self.parameters.current_dumps = self.parameters.current_dumps + 1;
        }

        // estimate of time left
        
    }

    /// This function computes the max timestep we can take, a constraint given by the minimum
    /// of the maximum kinetic, potential timesteps such that the wavefunction phase moves by >=2pi.
    pub fn get_timestep(&self, v: &Array<T>) -> (bool, T) {

        // Max kinetic
        let kgrid_max: T = get_kgrid::<T, S>(self.parameters.dx)
            .iter()
            .fold(T::zero(), |acc, x| if *x > acc { *x } else { acc });
        // Need to square and multiply by K to account for all k_i^2
        let kinetic_max: T = T::from_usize(K).unwrap() * kgrid_max * kgrid_max
            / (T::from_f64(2.0).unwrap() * self.parameters.particle_mass);
        let kinetic_dt: T = T::from_f64(2.0 * std::f64::consts::PI / HBAR).unwrap() / (kinetic_max);

        // Max potential  
        let potential_max: T = max_all(&v).0;
        let potential_dt: T = T::from_f64(2.0 * std::f64::consts::PI * HBAR).unwrap() / (potential_max);

        let time_to_next_dump = {
            let mut t = T::zero();
            while t < self.parameters.time {
                t = t + self.parameters.total_sim_time / T::from_u32(self.parameters.num_data_dumps).unwrap();
            }
            t - self.parameters.time
        };


        // least of three
        let dt = kinetic_dt.min(potential_dt).min(time_to_next_dump);

        let mut dump = false;
        if dt == time_to_next_dump {
            dump = true;
        } else if dt == kinetic_dt {
            println!("using kinetic {}", kinetic_dt);
        } else{
            println!("using potential {}, potential max is {}", potential_dt, potential_max);
        }
        (dump, dt)
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

        let rho = mul(
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
        );
        rho
    }

    /// This function calculates the potential for the stream
    pub fn calculate_potential(&self) -> Array<T> {

        let shape = self.get_shape().unwrap();

        // Compute space density and perform inplace fft
        let mut ρ: Array<Complex<T>> = self.get_density().cast();
        forward_inplace::<T, K, S>(&mut ρ);

        // Compute potential in k-space and perform inplace inverse fft
        let mut φ: Array<Complex<T>> = div(
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
            &self.get_spec_grid().cast(),
            false
        );
        let cond = isnan(&φ);
        let value = [false];
        let cond: Array<bool> = arrayfire::eq(&cond, &Array::new(&value, Dim4::new(&[1,1,1,1])), true);
        replace_scalar(&mut φ, &cond, 0.0);

        inverse_inplace::<T, K, S>(&mut φ);
        real(&φ)
    }

    /// This function writes out the wavefunction and metadata to disk
    pub fn dump(&self) {

        //
    }

    /// This function calculates k^2 grid
    pub fn get_spec_grid(&self) -> Array<T> {
        spec_grid::<T, K, S>(self.parameters.dx, self.get_shape().unwrap())
    }

    /// This function checks if simulation is done
    pub fn not_finished(&self) -> bool {
        self.parameters.time < self.parameters.total_sim_time
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

    const S: usize = 16;
    type T = f64;

    let axis_length: T = 1.0; 
    let time: T = 0.0;
    let total_sim_time: T = 1.0;
    let cfl: T = 0.25;
    let num_data_dumps: u32 = 100;
    let total_mass: T = 1.0;
    let particle_mass: T = 1e-12;
    let sim_name: &'static str = "my-sim";

    let params = SimulationParameters::<T,S>::new(
        axis_length,
        time,
        total_sim_time,
        cfl,
        num_data_dumps,
        total_mass,
        particle_mass,
        sim_name,
    );
    println!("{}", params);
}


