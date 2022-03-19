use arrayfire::{
    Array, ComplexFloating, HasAfEnum, FloatingPoint, ConstGenerator, Dim4, Fromf64,
    mul, real, conjg, constant, exp, max_all, div, replace_scalar, isnan, min_all, bitnot, sum_all, abs
};
use crate::{
    constants::{POIS_CONST, HBAR},
    utils::{
        grid::{check_complex_for_nans, check_for_nans},
        fft::{forward, inverse, forward_inplace, inverse_inplace, spec_grid, get_kgrid},
        complex::complex_constant,
        io::{array_to_disk, complex_array_to_disk},
        error::MSMError,
    }
};
use conv::*;
use std::fmt::Display;
use num::{Complex, Float, FromPrimitive};
use std::time::Instant;

/// This struct holds the grids which store the wavefunction, its Fourier transform, and other grids
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

    /// Potential
    pub φ: Array<T>,

    /// Fourier space
    pub spec_grid: Array<T>,

    /// Max of spec_grid_squared (k^2)
    pub k2_max: T,
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
    /// Current number of time steps
    pub time_steps: u64,
    /// Timestep Criterion
    pub cfl: U,

    // Physical Parameters
    /// Total Mass
    pub total_mass: f64,
    /// Particle mass
    pub particle_mass: f64,
    /// HBAR tilde (HBAR / particle_mass)
    pub hbar_: U,

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
    T: Float + FloatingPoint + ConstGenerator<OutType=T> + HasAfEnum<InType = T> + HasAfEnum<BaseType = T> + Fromf64 + FromPrimitive,
    Complex<T>: HasAfEnum + ComplexFloating + FloatingPoint + HasAfEnum<ComplexOutType = Complex<T>> + HasAfEnum<UnaryOutType = Complex<T>> + HasAfEnum<AbsOutType = T>,
 {

    pub fn new(
        ψ: Array<Complex<T>>, 
    ) -> Self
    {
        let ψk = forward::<T, K, S>(&ψ).expect("failed forward fft");
        let spec_grid =  spec_grid::<T, K, S>(T::one() / T::from_usize(S).unwrap(), 
            match K {
                1 => Ok((S as u64, 1, 1, 1)),
                2 => Ok((S as u64, S as u64, 1, 1)),
                3 => Ok((S as u64, S as u64, S as u64, 1)),
                _ => Err(MSMError::InvalidNumDumensions(K))
            }.unwrap()
        );
        let k2_max: T = max_all(&spec_grid).0;
        SimulationGrid {
            φ: real(&ψ).cast(), // TODO: change this to be 
            ψ,
            ψk,
            spec_grid,
            k2_max
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
        total_mass: f64,
        particle_mass: f64,
        sim_name: &'static str,

    ) -> Self
    {

        // Overconstrained or default parameters 
        let dx = axis_length / U::from_usize(S).unwrap();
        let dk = U::from_f64(2.0).unwrap() * U::from_f64(std::f64::consts::PI).unwrap() / axis_length;
        let current_dumps = 0;
        let hbar_ = U::from_f64(HBAR/particle_mass).unwrap();
        let time_steps = 0;

        SimulationParameters {
            axis_length,
            dx,
            dk,
            time,
            total_sim_time,
            cfl,
            num_data_dumps,
            current_dumps,
            time_steps,
            total_mass,
            particle_mass,
            hbar_,
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
    T: Float + FloatingPoint + Display + FromPrimitive + ConstGenerator<OutType=T> + HasAfEnum<InType = T> + HasAfEnum<AbsOutType = T> + HasAfEnum<AggregateOutType = T> + HasAfEnum<BaseType = T> + Fromf64 + ndarray_npy::WritableElement,
    Complex<T>: HasAfEnum + FloatingPoint + ComplexFloating + HasAfEnum<AggregateOutType = Complex<T>> + HasAfEnum<BaseType = T> + HasAfEnum<ComplexOutType = Complex<T>> + HasAfEnum<UnaryOutType = Complex<T>> + HasAfEnum<AbsOutType = T>,
{

    pub fn new(
        ψ: Array<Complex<T>>,
        axis_length: T,
        time: T,
        total_sim_time: T,
        cfl: T,
        num_data_dumps: u32,
        total_mass: f64,
        particle_mass: f64,
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

        // Calculate potential at t
        self.grid.φ = self.calculate_potential();
        debug_assert!(check_for_nans(&self.grid.φ));
        // Compute timestep
        let (dump, dt) = self.get_timestep();

        // Update kinetic half-step
        // exp(-(dt/2) * (k^2 / 2) / h) = exp(-dt/4/h * k^2)
        let k_evolution: Array<Complex<T>> = exp(
            &mul(
                &complex_constant(Complex::<T>::new(T::zero(), - dt / T::from_f64(4.0).unwrap() / self.parameters.hbar_), (1,1,1,1)),
                &self.grid.spec_grid.cast(),
                true
            )
        );
        // these are the fields with kinetic at t + dt/2 but momentum at t
        self.grid.ψk = mul(&self.grid.ψk, &k_evolution, false);
        debug_assert!(check_complex_for_nans(&self.grid.ψk));
        self.grid.ψ = inverse::<T, K, S>(&self.grid.ψk).unwrap();
        debug_assert!(check_complex_for_nans(&self.grid.ψ));
        self.grid.φ = self.calculate_potential();
        debug_assert!(check_for_nans(&self.grid.φ));

        // Update momentum a full-step
        // exp(-dt * φ / h) = exp(-(dt/h) * φ)
        let r_evolution: Array<Complex<T>> = exp(
            &mul(
                &complex_constant(Complex::<T>::new(T::zero(), -dt / self.parameters.hbar_), (1, 1, 1, 1)),
                &self.grid.φ.cast(),
                true
            )
        );
        // these are the fields with kinetic at t + dt/2 but momentum at t + dt
        self.grid.ψ = mul(&self.grid.ψ, &r_evolution, false);
        debug_assert!(check_complex_for_nans(&self.grid.ψ));
        self.grid.ψk = forward::<T, K, S>(&self.grid.ψ).unwrap();
        debug_assert!(check_complex_for_nans(&self.grid.ψk));


        // Update momentum from t + dt/2 to t + dt
        // exp(-(dt/2) * (k^2/2) / h) = exp(-dt/4/h * k^2)
        let k_evolution: Array<Complex<T>> = exp(
            &mul(
                &complex_constant(Complex::<T>::new(T::zero(), - dt / T::from_f64(4.0).unwrap() / self.parameters.hbar_), (1,1,1,1)),
                &self.grid.spec_grid.cast(),
                true
            )
        );
        // Now all fields have kinetic + momentum at t + dt
        self.grid.ψk = mul(&self.grid.ψk, &k_evolution, false);
        debug_assert!(check_complex_for_nans(&self.grid.ψk));
        self.grid.ψ = inverse::<T, K, S>(&self.grid.ψk).unwrap();
        debug_assert!(check_complex_for_nans(&self.grid.ψ));

        // Update time
        self.parameters.time = self.parameters.time + dt;

        // Print estimate of time to completion
        let estimate = now.elapsed().as_millis() * T::to_u128(&((self.parameters.total_sim_time - self.parameters.time)/dt)).unwrap_or(1);
        println!("update took {} millis, current sim time is {}, dt is {}. ETA {:?} ", now.elapsed().as_millis(), self.parameters.time, dt, std::time::Duration::from_millis(estimate as u64));

        if dump {
            self.dump();
            self.parameters.current_dumps = self.parameters.current_dumps + 1;
            self.parameters.time = T::from_u32(self.parameters.current_dumps).unwrap() * self.parameters.total_sim_time / T::from_u32(self.parameters.num_data_dumps).unwrap();
        }        
    }

    /// This function computes the max timestep we can take, a constraint given by the minimum
    /// of the maximum kinetic, potential timesteps such that the wavefunction phase moves by >=2pi.
    pub fn get_timestep(&self) -> (bool, T) {

        // Max kinetic dt
        // max(k^2)/2
        let kinetic_max: T = self.grid.k2_max / T::from_f64(2.0).unwrap();
        // dt = (2 * pi) / (hbar_ * max(k^2/2))
        let kinetic_dt: T = T::from_f64(2.0 * std::f64::consts::PI).unwrap() / (kinetic_max * self.parameters.hbar_);
        debug_assert!(kinetic_dt.is_finite(), "kinetic_dt is {}; hbar is {}",  kinetic_dt, self.parameters.hbar_);
        debug_assert!(kinetic_dt.is_sign_positive(),  "kinetic_dt is {}; hbar is {}", kinetic_dt, self.parameters.hbar_);
        debug_assert!(!kinetic_dt.is_zero(), "kinetic_dt is {}; hbar is {}", kinetic_dt, self.parameters.hbar_);

        // Max potential dt
        let potential_max: T = max_all(&abs(&self.grid.φ)).0;
        let potential_dt: T = T::from_f64(2.0 * std::f64::consts::PI).unwrap() * self.parameters.hbar_ / (potential_max);
        debug_assert!(potential_dt.is_finite());
        debug_assert!(potential_dt.is_sign_positive());
        debug_assert!(!potential_dt.is_zero());

        // Time to next data dump
        let time_to_next_dump = T::from_u32(self.parameters.current_dumps + 1).unwrap() * self.parameters.total_sim_time / T::from_u32(self.parameters.num_data_dumps).unwrap() - self.parameters.time; 

        // Take smallest of all time steps
        let dt = kinetic_dt.min(potential_dt).min(time_to_next_dump);

        // If taking time_to_next_dump, return dump flag
        let mut dump = false;
        if dt == time_to_next_dump { dump = true; println!("dump dt"); }

        // Return dump flag and timestep
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
    pub fn calculate_density(&self) -> Array<T> {

        let rho = mul(
            &Array::new(
                    &[T::from_f64(self.parameters.total_mass).unwrap()],
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
        let mut ρ: Array<Complex<T>> = self.calculate_density().cast();
        debug_assert!(check_complex_for_nans(&ρ));
        forward_inplace::<T, K, S>(&mut ρ);
        debug_assert!(check_complex_for_nans(&ρ));

        // Compute potential in k-space and perform inplace inverse fft
        let mut φ: Array<Complex<T>> = div(
            &mul(
                &Array::new(
                    &[Complex::<T>::new(
                        // TODO: check sign
                        T::from_f64(POIS_CONST).unwrap(),
                        T::zero()
                    )],
                    Dim4::new(&[1,1,1,1])
                ),
                &ρ,
                true
            ),
            &self.grid.spec_grid.cast(),
            false
        );

        // Populate 0 mode with 0.0
        let cond = isnan(&φ);
        let value = [false];
        let cond: Array<bool> = arrayfire::eq(&cond, &Array::new(&value, Dim4::new(&[1,1,1,1])), true);
        replace_scalar(&mut φ, &cond, 0.0);

        inverse_inplace::<T, K, S>(&mut φ);

        debug_assert!(check_complex_for_nans(&φ));

        real(&φ)
    }

    /// This function writes out the wavefunction and metadata to disk
    pub fn dump(&self) {

        let shape = self.get_shape().unwrap();
        // Dump psi
        complex_array_to_disk(
            format!("tests/data/psi_{}", self.parameters.current_dumps+1).as_str(),
            "psi",
            &self.grid.ψ,
            [shape.0, shape.1, shape.2, shape.3]
        ).expect("error writing psi to disk")
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


