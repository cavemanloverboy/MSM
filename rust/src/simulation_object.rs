use arrayfire::{
    Array, ComplexFloating, HasAfEnum, FloatingPoint, ConstGenerator, Dim4, Fromf64,
    mul, real, conjg, exp, max_all, div, replace_scalar, isnan, abs
};
use crate::{
    constants::{POIS_CONST, HBAR},
    utils::{
        grid::{check_complex_for_nans, check_for_nans, check_norm, Dimensions, IntoT},
        fft::{forward, inverse, forward_inplace, inverse_inplace, spec_grid},
        complex::complex_constant,
        io::{complex_array_to_disk, read_toml},
        error::*,
    },
    ics::{InitialConditions, *},
};
use ndarray_npy::{ReadableElement, WritableElement};
use anyhow::{Result, Context, bail};
use std::fmt::Display;
use num::{Complex, Float, FromPrimitive, ToPrimitive};
use std::time::Instant;
use serde_derive::Deserialize;
use num_traits::FloatConst;


#[derive(Deserialize)]
struct TomlParameters {
    
    // Required Parameters
    /// Physical length of box
    pub axis_length: f64,
    /// Start (and current) time of simulation
    pub time: Option<f64>,
    /// End time of simulation
    pub final_sim_time: f64,
    /// Safety factor
    pub cfl: f64,
    /// Number of data dumps
    pub num_data_dumps: u32,
    /// Total mass in the system
    pub total_mass: f64,
    /// Particle mass (use this or hbar_ = HBAR / particle_mass)
    pub particle_mass: Option<f64>,
    /// HBAR / particle mass (use this or particle_mass)
    pub hbar_: Option<f64>,
    /// Name of simulation (used for directories)
    pub sim_name: String,
    /// Where to begin checking for cutoff (this is a parameter in [0,1])
    pub k2_cutoff: f64,
    /// How much of mass to use as threshold for constitutes aliasing (P(k > k2_cutoff * k2))
    pub alias_threshold: f64,
    /// Dimensionality of grid
    pub dims: usize,
    /// Number of grid cells per dim
    pub size: usize,
    /// Initial Conditions
    pub ics: InitialConditions,
}


/// This struct holds the grids which store the wavefunction, its Fourier transform, and other grids
pub struct SimulationGrid<T>
where
    T: Float + FloatingPoint + ConstGenerator<OutType=T> + HasAfEnum<InType = T> + HasAfEnum<BaseType = T> + FromPrimitive,
    Complex<T>: HasAfEnum + FloatingPoint + HasAfEnum<AbsOutType = T> + ConstGenerator<OutType=Complex<T>>,
{

    // Spatial Fields
    /// The array which stores the wavefunction
    pub ψ: Array<Complex<T>>,

    /// Fourier space
    pub ψk: Array<Complex<T>>,

    /// Potential
    pub φ: Array<Complex<T>>,
}



/// This `Parameters` struct stores simulations parameters
#[derive(Clone)]
pub struct SimulationParameters<T: Float + FloatingPoint> {

    // Grid Parameters
    /// Physical length of each axis
    pub axis_length: T,
    /// Spatial cell size
    pub dx: T,
    /// k-space cell size
    pub dk: T,
    /// Fourier grid (k^2)
    pub spec_grid: Array<T>,
    /// Max of Fourier grid
    pub k2_max: T,
    /// Dimensionality of grid
    pub dims: Dimensions,
    /// Number of grid cells
    pub size: usize,

    // Temporal Parameters
    /// Current simulation time
    pub time: T,
    /// Total simulation time
    pub final_sim_time: T,
    /// Total number of data dumps
    pub num_data_dumps: u32,
    /// Current number of data dumps
    pub current_dumps: u32,
    /// Current number of time steps
    pub time_steps: u64,
    /// Timestep Criterion
    pub cfl: T,

    // Physical Parameters
    /// Total Mass
    pub total_mass: f64,
    /// Particle mass
    pub particle_mass: f64,
    /// HBAR tilde (HBAR / particle_mass)
    pub hbar_: T,
    /// Total number of particles
    pub n_tot: f64,

    // Simulation parameters and metadata
    /// Simulation name
    pub sim_name: String,
    /// Fourier alias bound, in [0, 1]
    pub k2_cutoff: f64,
    /// Alias threshold (probability mass), in [0,1]
    pub alias_threshold: f64,
    /// Simulation wall time (millis)
    pub sim_wall_time: u128,
    /// Number of timesteps taken
    pub n_steps: u64,
    
}

/// In the original python implementation, this was a `sim` or `SimObject` object.
/// This stores a `SimulationGrid` which has the wavefunction and its fourier transform.
/// It also holds the `SimulationParameters` which holds the simulation parameters.
pub struct SimulationObject<T>
where
    T: Float + FloatingPoint + ConstGenerator<OutType=T> + HasAfEnum<InType = T> + HasAfEnum<BaseType = T> + FromPrimitive + std::fmt::LowerExp + FloatConst,
    Complex<T>: HasAfEnum + ComplexFloating + FloatingPoint + HasAfEnum<AbsOutType = T> + HasAfEnum<ArgOutType = T> + ConstGenerator<OutType=Complex<T>>,
{

    /// This has the wavefunction and its Fourier transform
    pub grid: SimulationGrid<T>,

    /// This has the simulation parameters
    pub parameters: SimulationParameters<T>,

}

impl<T> SimulationGrid<T>
where
    T: Float + FloatingPoint + ConstGenerator<OutType=T> + HasAfEnum<InType = T> + HasAfEnum<BaseType = T> + FromPrimitive,
    Complex<T>: HasAfEnum + ComplexFloating + FloatingPoint + HasAfEnum<ComplexOutType = Complex<T>> + HasAfEnum<UnaryOutType = Complex<T>> + HasAfEnum<AbsOutType = T> + HasAfEnum<ArgOutType = T> + ConstGenerator<OutType=Complex<T>> ,
 {

    pub fn new(
        ψ: Array<Complex<T>>,
    ) -> Self
    {
        SimulationGrid {
            φ: real(&ψ).cast(), // Note: Initialized with incorrect values!
            ψk: ψ.clone(), // Note: Initialize with incorrect values
            ψ,
        }
    }

}

impl<T> SimulationParameters<T>
where
    T: FromPrimitive + Float + FloatingPoint + Display + HasAfEnum<InType = T> + HasAfEnum<BaseType = T> + Fromf64
{

    pub fn new(
        axis_length: T,
        time: T,
        final_sim_time: T,
        cfl: T,
        num_data_dumps: u32,
        total_mass: f64,
        particle_mass: f64,
        sim_name: String,
        k2_cutoff: f64,
        alias_threshold: f64,
        hbar_: Option<f64>,
        dims: Dimensions,
        size: usize,
    ) -> Self
    {

        // Overconstrained or default parameters 
        let dx = axis_length / T::from_usize(size).unwrap();
        //let dk = U::from_f64(2.0).unwrap() * U::from_f64(std::f64::consts::PI).unwrap() / axis_length;
        let dk = dx; //TODO: figure out why thiis works //U::one() / axis_length / U::from_usize(S).unwrap();
        //let dk = get_kgrid::<U, S>(dx)[1];
        let current_dumps = 0;

        let hbar_: T = T::from_f64(hbar_.unwrap_or(HBAR / particle_mass)).unwrap();
        let time_steps = 0;

        let spec_grid = spec_grid::<T>(dx, dims, size);

        let k2_max: T = max_all(&spec_grid).0;
        let sim_wall_time = 0;

        let n_tot = total_mass / particle_mass;
        let n_steps = 0;

        SimulationParameters {
            axis_length,
            dx,
            dk,
            time,
            final_sim_time,
            cfl,
            num_data_dumps,
            current_dumps,
            time_steps,
            total_mass,
            particle_mass,
            hbar_,
            sim_name, 
            k2_cutoff,
            alias_threshold,
            spec_grid,
            k2_max,
            sim_wall_time,
            n_tot,
            size,
            dims,
            n_steps,
        }
    }

    pub fn get_shape(&self) -> [u64; 4] {
        match self.dims {
            Dimensions::One => [self.size as u64, 1, 1, 1],
            Dimensions::Two => [self.size as u64, self.size as u64, 1, 1],
            Dimensions::Three => [self.size as u64, self.size as u64, self.size as u64, 1],
        }
    }
}
impl<U> Display for SimulationParameters<U>
where
    U: Float + FloatingPoint + Display
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}\n","-".repeat(40))?;
        write!(f, "axis_length         = {}\n", self.axis_length)?;
        write!(f, "dx                  = {}\n", self.dx)?;
        write!(f, "current_time        = {}\n", self.time)?;
        write!(f, "final_sim_time      = {}\n", self.final_sim_time)?;
        write!(f, "cfl                 = {}\n", self.cfl)?;
        write!(f, "num_data_dumps      = {}\n", self.num_data_dumps)?;
        write!(f, "total_mass          = {}\n", self.total_mass)?;
        write!(f, "particle_mass       = {}\n", self.particle_mass)?;
        write!(f, "hbar_               = {}\n", self.hbar_)?;
        write!(f, "sim_name            = {}\n", self.sim_name)?;
        write!(f, "k2_cutoff           = {}\n", self.k2_cutoff)?;
        write!(f, "alias_threshold     = {}\n", self.alias_threshold)?;
        write!(f, "k2_max              = {}\n", self.k2_max)?;
        write!(f, "n_tot               = {}\n", self.n_tot)?;
        write!(f, "dims                = {}\n", self.dims as usize)?;
        write!(f, "size                = {}\n", self.size as usize)?;
        write!(f, "{}\n","-".repeat(40))?;
        Ok(())
    }
}



impl<T> SimulationObject<T>
where
    T: Float + FloatingPoint + Display + ToPrimitive + FromPrimitive + ConstGenerator<OutType=T> + HasAfEnum<InType = T> + HasAfEnum<AbsOutType = T> + HasAfEnum<AggregateOutType = T> + HasAfEnum<BaseType = T> + Fromf64 + WritableElement + ReadableElement + std::fmt::LowerExp + FloatConst,
    Complex<T>: HasAfEnum + FloatingPoint + ComplexFloating + HasAfEnum<AggregateOutType = Complex<T>> + HasAfEnum<BaseType = T> + HasAfEnum<ComplexOutType = Complex<T>> + HasAfEnum<UnaryOutType = Complex<T>> + HasAfEnum<AbsOutType = T> + HasAfEnum<ArgOutType = T> + ConstGenerator<OutType=Complex<T>>,
    rand_distr::Standard: rand_distr::Distribution<T>
{
    /// A constructor function which returns a `SimulationObject`
    pub fn new(
        ψ: Array<Complex<T>>,
        axis_length: T,
        time: T,
        final_sim_time: T,
        cfl: T,
        num_data_dumps: u32,
        total_mass: f64,
        particle_mass: f64,
        sim_name: String,
        k2_cutoff: f64,
        alias_threshold: f64,
        hbar_: Option<f64>,
        dims: Dimensions,
        size: usize,
    ) -> Self {
        
        // Construct components
        let grid = SimulationGrid::<T>::new(ψ);
        let parameters = SimulationParameters::<T>::new(
            axis_length,
            time,
            final_sim_time,
            cfl,
            num_data_dumps,
            total_mass,
            particle_mass,
            sim_name,
            k2_cutoff,
            alias_threshold,
            hbar_,
            dims,
            size,
        );

        let sim_obj = SimulationObject {
            grid,
            parameters,
        };
        debug_assert!(check_norm::<T>(&sim_obj.grid.ψ, sim_obj.parameters.dx, dims));
        debug_assert!(check_norm::<T>(&sim_obj.grid.ψk, sim_obj.parameters.dk, dims));
        sim_obj
    }

    /// A constructor function which returns a `SimulationObject`
    pub fn new_with_parameters(
        ψ: Array<Complex<T>>,
        parameters: SimulationParameters<T>,
    ) -> Self {
        
        // Construct components
        let grid = SimulationGrid::<T>::new(ψ);

        let sim_obj = SimulationObject {
            grid,
            parameters,
        };
        debug_assert!(check_norm::<T>(&sim_obj.grid.ψ, sim_obj.parameters.dx, sim_obj.parameters.dims));
        debug_assert!(check_norm::<T>(&sim_obj.grid.ψk, sim_obj.parameters.dk, sim_obj.parameters.dims));
        sim_obj
    }

    /// A constructor function which returns a `SimulationObject` from a user's toml.
    pub fn new_from_toml(
        path: String,
    ) -> Self {

        let toml: TomlParameters = read_toml(path);

        // Required Parameters
        let axis_length: T = T::from_f64(toml.axis_length).unwrap();
        let time: T = T::from_f64(toml.time.unwrap_or(0.0)).unwrap();
        let final_sim_time: T = T::from_f64(toml.final_sim_time).unwrap();
        let cfl: T = T::from_f64(toml.cfl).unwrap();
        let num_data_dumps: u32 = toml.num_data_dumps;
        let total_mass: f64 = toml.total_mass;
        let sim_name: String = toml.sim_name;
        let k2_cutoff: f64 = toml.k2_cutoff;
        let alias_threshold: f64 = toml.alias_threshold;
        let dims = num::FromPrimitive::from_usize(toml.dims).unwrap();
        let size = toml.size;

        // Overdetermined
        let particle_mass;
        let hbar_;
        if toml.particle_mass.is_some() {
            assert!(toml.hbar_.is_none(), "Cannot specify hbar_ and particle_mass");
            particle_mass = toml.particle_mass.unwrap();
            hbar_ = HBAR / particle_mass;
        } else {
            assert!(toml.hbar_.is_some(), "Must specify hbar_ or particle_mass");
            hbar_ = toml.hbar_.unwrap();
            particle_mass = HBAR / hbar_ ;
        }

        // Construct `SimulationParameters`
        let mut parameters = SimulationParameters::<T>::new(
            axis_length,
            time,
            final_sim_time,
            cfl,
            num_data_dumps,
            total_mass,
            particle_mass,
            sim_name,
            k2_cutoff,
            alias_threshold,
            Some(hbar_),
            dims,
            size,
        );

        // Construct wavefunction
        let grid: SimulationGrid::<T> = match toml.ics {
            
            // User-specified Initial Conditions
            InitialConditions::UserSpecified { path } 
                => SimulationGrid::<T>::new(user_specified_ics(path, &mut parameters)),

            // Real space gaussian 
            InitialConditions::ColdGaussMFT{ mean, std} 
                => cold_gauss::<T>(mean.intoT(), std.intoT(), &parameters).grid,
            InitialConditions::ColdGaussMSM{ mean, std, scheme, sample_seed } 
                => cold_gauss_sample::<T>(mean.intoT(), std.intoT(), &parameters, scheme, sample_seed).grid,

            // Momentum space gaussian
            InitialConditions::ColdGaussKSpaceMFT{ mean, std, phase_seed} 
                => cold_gauss_kspace::<T>(mean.intoT(), std.intoT(), &parameters, phase_seed).grid,
            InitialConditions::ColdGaussKSpaceMSM{ mean, std, scheme, phase_seed, sample_seed } 
                => cold_gauss_kspace_sample::<T>(mean.intoT(), std.intoT(), &parameters, scheme, phase_seed, sample_seed).grid,
            
            // Spherical Tophat
            InitialConditions::SphericalTophat{ radius, delta, slope }
                => spherical_tophat::<T>(&parameters, radius, delta, slope).grid,

            _ => todo!("You must have passed an enum for a set of ics that is not yet implemented"),
        };


        // Pack grid and parameters into `Simulation Object`
        let sim_obj = SimulationObject {
            grid,
            parameters,
        };
        debug_assert!(check_norm::<T>(&sim_obj.grid.ψ, sim_obj.parameters.dx, dims));
        debug_assert!(check_norm::<T>(&sim_obj.grid.ψk, sim_obj.parameters.dk, dims));

        sim_obj
    }


    /// This function updates the `SimulationGrid` stored in the `SimulationObject`.
    pub fn update(&mut self, verbose: bool) -> Result<()> {

        // If this is the first timestep, populate the kspace grid with the correct values
        if self.parameters.n_steps == 0 {
            println!("Initializing k-space wavefunction");
            self.grid.ψk = self.forward(&self.grid.ψ).unwrap();
        };

        // Begin timer for update loop
        let now = Instant::now();

        // Initial checks
        debug_assert!(check_norm::<T>(&self.grid.ψ, self.parameters.dx, self.parameters.dims));
        debug_assert!(check_norm::<T>(&self.grid.ψk, self.parameters.dk, self.parameters.dims));

        // Calculate potential at t
        self.calculate_potential();
        debug_assert!(check_complex_for_nans(&self.grid.φ));
        // Compute timestep
        let (dump, dt) = self.get_timestep();

        // Update kinetic half-step
        // exp(-(dt/2) * (k^2 / 2) / h_) = exp(-dt/4/h_ * k^2)
        let k_evolution: Array<Complex<T>> = exp(
            &mul(
                &complex_constant(Complex::<T>::new(T::zero(), - dt / T::from_f64(4.0).unwrap() * self.parameters.hbar_), (1,1,1,1)),
                &self.parameters.spec_grid.cast(),
                true
            )
        );
        // These are the fields with kinetic at t + dt/2 but momentum at t
        self.grid.ψk = mul(&self.grid.ψk, &k_evolution, false);
        debug_assert!(check_complex_for_nans(&self.grid.ψk));
        debug_assert!(check_norm::<T>(&self.grid.ψk, self.parameters.dk, self.parameters.dims));
        self.grid.ψ = self.inverse(&self.grid.ψk).unwrap();
        debug_assert!(check_complex_for_nans(&self.grid.ψ));
        debug_assert!(check_norm::<T>(&self.grid.ψ, self.parameters.dx, self.parameters.dims));
        self.calculate_potential();
        debug_assert!(check_complex_for_nans(&self.grid.φ));

        // Update momentum a full-step
        // exp(-dt * φ / h_) = exp(-(dt/h_) * φ)
        let r_evolution: Array<Complex<T>> = exp(
            &mul(
                &complex_constant(Complex::<T>::new(T::zero(), -dt / self.parameters.hbar_), (1, 1, 1, 1)),
                &self.grid.φ.cast(),
                true
            )
        );
        //complex_array_to_disk("r_evo", "r_evo", &r_evolution, [shape.0, shape.1, shape.2, shape.3]);
        // these are the fields with kinetic at t + dt/2 but momentum at t + dt
        self.grid.ψ = mul(&self.grid.ψ, &r_evolution, false);
        debug_assert!(check_complex_for_nans(&self.grid.ψ));
        debug_assert!(check_norm::<T>(&self.grid.ψ, self.parameters.dx, self.parameters.dims));
        self.grid.ψk = self.forward(&self.grid.ψ).unwrap();
        debug_assert!(check_complex_for_nans(&self.grid.ψk));
        debug_assert!(check_norm::<T>(&self.grid.ψk, self.parameters.dk, self.parameters.dims));


        // Update momentum from t + dt/2 to t + dt
        // exp(-(dt/2) * (k^2/2) / h) = exp(-dt/4/h * k^2)
        let k_evolution: Array<Complex<T>> = exp(
            &mul(
                &complex_constant(Complex::<T>::new(T::zero(), - dt / T::from_f64(4.0).unwrap() * self.parameters.hbar_), (1,1,1,1)),
                &self.parameters.spec_grid.cast(),
                true
            )
        );
        //complex_array_to_disk("k_evo", "k_evo", &k_evolution, [shape.0, shape.1, shape.2, shape.3]);
        //assert!(false);
        // Now all fields have kinetic + momentum at t + dt
        self.grid.ψk = mul(&self.grid.ψk, &k_evolution, false);
        debug_assert!(check_complex_for_nans(&self.grid.ψk));
        debug_assert!(check_norm::<T>(&self.grid.ψk, self.parameters.dk, self.parameters.dims));
        self.grid.ψ = self.inverse(&self.grid.ψk)?;
        debug_assert!(check_complex_for_nans(&self.grid.ψ));
        debug_assert!(check_norm::<T>(&self.grid.ψ, self.parameters.dx, self.parameters.dims));

        // Update time
        self.parameters.time = self.parameters.time + dt;

        // Print estimate of time to completion
        let estimate = now.elapsed().as_millis() * T::to_u128(&((self.parameters.final_sim_time - self.parameters.time)/dt)).unwrap_or(1);
        if verbose {println!("update took {} millis, current sim time is {:e}, dt is {:e}. ETA {:?} ", now.elapsed().as_millis(), self.parameters.time, dt, std::time::Duration::from_millis(estimate as u64));}

        // Check for Fourier Aliasing
        let aliased = self.check_alias();
        if aliased.is_some() {
            println!("currently aliased!");
            // If above threshold, bail and report aliasing
            bail!(RuntimeError::FourierAliasing {
                threshold: self.parameters.alias_threshold as f32,
                k2_cutoff: self.parameters.k2_cutoff as f32,
                p_mass: T::to_f32(&aliased.unwrap()).unwrap()
            });
        }

        // Perform data dump if appropriate
        if dump {
            self.dump();
            self.parameters.time = T::from_u32(self.parameters.current_dumps).unwrap() * self.parameters.final_sim_time / T::from_u32(self.parameters.num_data_dumps).unwrap();
        }        

        // Increment wall time counter, step counter
        self.parameters.sim_wall_time += now.elapsed().as_millis();
        self.parameters.n_steps += 1;
        
        Ok(())
    }

    /// This function computes the max timestep we can take, a constraint given by the minimum
    /// of the maximum kinetic, potential timesteps such that the wavefunction phase moves by >=2pi.
    pub fn get_timestep(&self) -> (bool, T) {

        // Max kinetic dt
        // max(k^2)/2
        let kinetic_dt: T = self.parameters.cfl * T::from_f64(2.0).unwrap() * self.parameters.axis_length / self.parameters.k2_max.sqrt() / self.parameters.hbar_;
        debug_assert!(kinetic_dt.is_finite(), "kinetic_dt is {}; hbar_ is {}",  kinetic_dt, self.parameters.hbar_);
        debug_assert!(kinetic_dt.is_sign_positive(),  "kinetic_dt is {}; hbar_ is {}", kinetic_dt, self.parameters.hbar_);
        debug_assert!(!kinetic_dt.is_zero(), "kinetic_dt is {}; hbar_ is {}", kinetic_dt, self.parameters.hbar_);

        // Max potential dt
        let potential_max: T = max_all(&abs(&self.grid.φ)).0;
        let potential_dt: T = self.parameters.cfl * T::from_f64(2.0 * std::f64::consts::PI).unwrap() * self.parameters.hbar_ / ( T::from_f64(2.0).unwrap() * potential_max );
        debug_assert!(potential_dt.is_finite());
        debug_assert!(potential_dt.is_sign_positive());
        debug_assert!(!potential_dt.is_zero());

        // Time to next data dump
        let time_to_next_dump = T::from_u32(self.parameters.current_dumps + 1).unwrap() * self.parameters.final_sim_time / T::from_u32(self.parameters.num_data_dumps).unwrap() - self.parameters.time; 

        // Take smallest of all time steps
        let dt = kinetic_dt.min(potential_dt).min(time_to_next_dump);
        // println!("kinetic = {:.4e}, potential = {:.4e} kinetic/potential = {}", kinetic_dt, potential_dt, kinetic_dt/potential_dt);

        // If taking time_to_next_dump, return dump flag
        let mut dump = false;
        if dt == time_to_next_dump { dump = true; }// println!("dump dt"); }

        // Return dump flag and timestep
        (dump, dt)
    }
    
    /// This function computes the shape of the grid
    pub fn get_shape(&self) -> (u64, u64, u64, u64) {
        match self.parameters.dims {
            Dimensions::One => (self.parameters.size as u64, 1, 1, 1),
            Dimensions::Two => (self.parameters.size as u64, self.parameters.size as u64, 1, 1),
            Dimensions::Three => (self.parameters.size as u64, self.parameters.size as u64, self.parameters.size as u64, 1),
        }
    }

    // This function computes the shape of the grid
    pub fn get_shape_array(&self) -> [u64; 4] {
        match self.parameters.dims {
            Dimensions::One => [self.parameters.size as u64, 1, 1, 1],
            Dimensions::Two => [self.parameters.size as u64, self.parameters.size as u64, 1, 1],
            Dimensions::Three => [self.parameters.size as u64, self.parameters.size as u64, self.parameters.size as u64, 1],
        }
    }

    /// This function computes the space density 
    pub fn calculate_density(&mut self) {

        // We reuse the memory for φ
        self.grid.φ = mul(
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
        ).cast();
    }

    /// This function calculates the potential for the stream
    pub fn calculate_potential(&mut self) {

        // Compute space density and perform inplace fft
        // note: this is using memory location of self.grid.φ
        self.calculate_density();
        debug_assert!(check_complex_for_nans(&self.grid.φ));
        self.forward_potential_inplace().expect("failed to do forward fft for potential");
        debug_assert!(check_complex_for_nans(&self.grid.φ));

        // Compute potential in k-space and perform inplace inverse fft
        self.grid.φ = div(
            &mul(
                &Array::new(
                    &[Complex::<T>::new(
                        // TODO: check sign
                        T::from_f64(-POIS_CONST).unwrap(),
                        T::zero()
                    )],
                    Dim4::new(&[1,1,1,1])
                ),
                &
                self.grid.φ,
                true
            ),
            &self.parameters.spec_grid.cast(),
            false
        );


        // Populate 0 mode with 0.0
        let cond = isnan(&self.grid.φ);
        let value = [false];
        let cond: Array<bool> = arrayfire::eq(&cond, &Array::new(&value, Dim4::new(&[1,1,1,1])), true);
        replace_scalar(&mut self.grid.φ, &cond, 0.0);

        self.inverse_potential_inplace().expect("failed to do inverse fft for potential");

        debug_assert!(check_complex_for_nans(&self.grid.φ));

        self.grid.φ = real(&self.grid.φ).cast();
    }

    /// This function writes out the wavefunction and metadata to disk
    pub fn dump(&mut self) {

        let shape = self.get_shape_array();

        // Create directory if necessary
        std::fs::create_dir_all(format!("sim_data/{}/", self.parameters.sim_name)).expect("failed to make directory");

        // Dump psi
        complex_array_to_disk(
            format!("sim_data/{}/psi_{:05}", self.parameters.sim_name, self.parameters.current_dumps).as_str(),
            "psi",
            &self.grid.ψ,
            shape
        ).context(RuntimeError::IOError).unwrap();
        
        self.parameters.current_dumps = self.parameters.current_dumps + 1;
    }

    /// This function checks if simulation is done
    pub fn not_finished(&self) -> bool {
        self.parameters.time < self.parameters.final_sim_time
    }


    /// This function outputs a text file
    pub fn dump_parameters(&self, additional_parameters: Vec<String>) {

        // Create directory if necessary
        std::fs::create_dir_all(format!("sim_data/{}/", self.parameters.sim_name)).expect("failed to make directory");

        // Location of parameter file
        let param_file: String = format!("sim_data/{}/parameters.txt", self.parameters.sim_name);

        // Write to parameter file
        std::fs::write(param_file, format!("{}{}", self.parameters, additional_parameters.join(""))).expect("Failed to write parameter file");
    }

    /// This function checks the Fourier space wavefunction for aliasing
    pub fn check_alias(&self) -> Option<T> {
        
        // Clone the Fourier space wavefunction
        let alias_check = self.grid.ψk.copy();
        debug_assert!(crate::utils::grid::check_norm::<T>(&alias_check, self.parameters.dk, self.parameters.dims));

        // Norm squared, cast to real
        let mut alias_check: Array<T> = real(&mul(
            &alias_check,
            &conjg(&alias_check),
            false
        ));

        // Replace all values under cutoff with 0
        let is_over_cutoff = arrayfire::gt(
            &self.parameters.spec_grid, 
            &arrayfire::constant(
                self.parameters.k2_max * T::from_f64(self.parameters.k2_cutoff).unwrap(),
                Dim4::new(&[1, 1, 1, 1]),
            ),
            true
        ); 
        replace_scalar::<T>(
            // Array to replace
            &mut alias_check, 
            // Condition to check for
            &is_over_cutoff,
            // Value to replace with when false
            0.0
        );

        // Sum all remaining values (those over cutoff) to get total mass that is near-aliasing
        let sum = arrayfire::sum_all(
            &alias_check
        );
        let p_mass = sum.0*self.parameters.dk.powf(T::from_usize(self.parameters.dims as usize).unwrap());

        // If above threshold, return Some. Otherwise, return None
        if p_mass > T::from_f64(self.parameters.alias_threshold).unwrap() {
            Some(p_mass)
        } else {
            None
        }
    }


    fn forward_inplace(&mut self, array: &mut Array<Complex<T>>) -> Result<()> {
        let dims = self.parameters.dims;
        let size = self.parameters.size;
        forward_inplace(array, dims, size)
    }

    fn inverse_inplace(&mut self, array: &mut Array<Complex<T>>) -> Result<()> {
        let dims = self.parameters.dims;
        let size = self.parameters.size;
        inverse_inplace(array, dims, size)
    }

    fn forward_potential_inplace(&mut self) -> Result<()> {
        let dims = self.parameters.dims;
        let size = self.parameters.size;
        forward_inplace(&mut self.grid.φ, dims, size)
    }

    fn inverse_potential_inplace(&mut self) -> Result<()> {
        let dims = self.parameters.dims;
        let size = self.parameters.size;
        inverse_inplace(&mut self.grid.φ, dims, size)
    }

    fn forward(&self, array: &Array<Complex<T>>) -> Result<Array<Complex<T>>> {
        let dims = self.parameters.dims;
        let size = self.parameters.size;
        forward(array, dims, size)
    }

    fn inverse(&self, array: &Array<Complex<T>>) -> Result<Array<Complex<T>>> {
        let dims = self.parameters.dims;
        let size = self.parameters.size;
        inverse(array, dims, size)
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
    let _grid: SimulationGrid<f32> = SimulationGrid::<f32>::new(ψ);
    //af_print!("ψ", grid.ψ);
    //af_print!("ψk", grid.ψk);
}


#[test]
fn test_new_sim_parameters() {

    type T = f64;
    let dims = Dimensions::One;
    let size = 16;

    let axis_length: T = 1.0; 
    let time: T = 0.0;
    let final_sim_time: T = 1.0;
    let cfl: T = 0.25;
    let num_data_dumps: u32 = 100;
    let total_mass: T = 1.0;
    let particle_mass: T = 1e-12;
    let sim_name: String = "my-sim".to_string();
    let k2_cutoff: f64 = 0.95;
    let alias_threshold: f64 = 0.02;
    let hbar_ = None;

    let params = SimulationParameters::<T>::new(
        axis_length,
        time,
        final_sim_time,
        cfl,
        num_data_dumps,
        total_mass,
        particle_mass,
        sim_name,
        k2_cutoff,
        alias_threshold,
        hbar_,
        dims,
        size,
    );
    println!("{}", params);
}


#[test]
fn test_lt_gt() {

    type T = f32;
    const S: usize = 16;
    let values1: [T; S] = [5.0; S];
    let values2: [T; S] = [4.0; S];

    let mut array1 = arrayfire::Array::new(&values1, Dim4::new(&[S as u64, 1, 1, 1]));
    let array2 = arrayfire::Array::new(&values2, Dim4::new(&[S as u64, 1, 1, 1]));

    let is_under = arrayfire::lt(
        &array1.clone(),
        &array2,   
        false
    );

    replace_scalar::<T>(
        // Array to replace
        &mut array1, 
        // Condition to check for
        &is_under,
        // Value to replace with if true
        1e2
    );

    println!("gt sum is {}", arrayfire::sum_all(&array1).0);
    println!("lt sum is {}", arrayfire::sum_all(&array1).0);

}
