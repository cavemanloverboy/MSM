use crate::{
    simulation_object::*,
    utils::{
        grid::{normalize, check_norm, Dimensions},
        complex::complex_constant,
        fft::{forward_inplace, get_kgrid},
    },
};
use arrayfire::{Array, ComplexFloating, HasAfEnum, FloatingPoint, Dim4, add, mul, exp, random_uniform, conjg, arg, div, abs, Fromf64, ConstGenerator, RandomEngine};
use num::{Complex, Float, FromPrimitive, ToPrimitive};
use ndarray::OwnedRepr;
use ndarray_npy::{WritableElement, ReadableElement};
use num_traits::FloatConst;
use std::fmt::Display;
use std::iter::Iterator;
use rand_distr::{Poisson, Distribution};
use serde_derive::{Serialize, Deserialize};

#[derive(Serialize, Deserialize)]
pub enum InitialConditions {
    UserSpecified {
        path: String
    },
    ColdGaussMFT {
        mean: Vec<f64>,
        std: Vec<f64>
    },
    ColdGaussMSM {
        mean: Vec<f64>,
        std: Vec<f64>,
        scheme: SamplingScheme,
        phase_seed: Option<u64>,
        sample_seed: Option<u64>,
    },
}

/// This function produces initial conditions corresonding to a cold initial gaussian in sp
pub fn cold_gauss<T>(
    mean: Vec<T>,
    std: Vec<T>,
    parameters: &SimulationParameters<T>,
) -> SimulationObject<T>
where
    T: Float + FloatingPoint + FromPrimitive + Display + Fromf64 + ConstGenerator<OutType=T> + HasAfEnum<AggregateOutType = T> + HasAfEnum<InType = T> + HasAfEnum<AbsOutType = T> + HasAfEnum<BaseType = T> + Fromf64 + WritableElement + ReadableElement + std::fmt::LowerExp + FloatConst,
    Complex<T>: HasAfEnum + ComplexFloating + FloatingPoint + HasAfEnum<ComplexOutType = Complex<T>> + HasAfEnum<UnaryOutType = Complex<T>> + HasAfEnum<AggregateOutType = Complex<T>> + HasAfEnum<AbsOutType = T>  + HasAfEnum<BaseType = T> + HasAfEnum<ArgOutType = T> + ConstGenerator<OutType=Complex<T>>,
    rand_distr::Standard: Distribution<T>,
{
    assert_eq!(mean.len(), parameters.dims as usize, "Cold Gauss: Mean vector provided has incorrect dimensionality");
    assert_eq!(std.len(), parameters.dims as usize, "Cold Gauss: Std vector provided has incorrect dimensionality");

    // Construct spatial grid
    let x: Vec<T> = (0..parameters.size)
        .map(|i| T::from_usize(i).unwrap() * parameters.dx)
        .collect();
    let y = &x;
    let z = &x;

    // Construct ψx
    let mut ψx_values = vec![Complex::<T>::new(T::zero(), T::zero()); parameters.size];
    for (i, ψx_val) in ψx_values.iter_mut().enumerate(){
        *ψx_val = Complex::<T>::new(
            (T::from_f64(-0.5).unwrap() * ((x[i] - mean[0]) / std[0]).powf(T::from_f64(2.0).unwrap())).exp(),
            T::zero(),
        );
    }
    let x_dims = Dim4::new(&[parameters.size as u64, 1, 1, 1]);
    let mut ψx: Array<Complex<T>> = Array::new(&ψx_values, x_dims);
    normalize::<T>(&mut ψx, parameters.dx, parameters.dims);
    debug_assert!(check_norm::<T>(&ψx, parameters.dx, parameters.dims));

    // Construct ψy
    let mut ψy;
    if parameters.dims as usize >= 2 {
        let mut ψy_values = vec![Complex::<T>::new(T::zero(), T::zero()); parameters.size];
        for (i, ψy_val) in ψy_values.iter_mut().enumerate(){
            *ψy_val = Complex::<T>::new(
                (T::from_f64(-0.5).unwrap() * ((y[i] - mean[1]) / std[1]).powf(T::from_f64(2.0).unwrap())).exp(),
                T::zero(),
            );
        }

        let y_dims = Dim4::new(&[1, parameters.size as u64, 1, 1]);
        ψy = Array::new(&ψy_values, y_dims);
        normalize::<T>(&mut ψy, parameters.dx, parameters.dims);
        debug_assert!(check_norm::<T>(&ψy, parameters.dx, parameters.dims));
    } else {
        let y_dims = Dim4::new(&[1, 1, 1, 1]);
        ψy = Array::new(&[Complex::<T>::new(T::one(), T::zero())], y_dims);
    }



    // Construct ψz
    let mut ψz;
    if parameters.dims as usize == 3 {
        let mut ψz_values = vec![Complex::<T>::new(T::zero(), T::zero()); parameters.size];
        for (i, ψz_val) in ψz_values.iter_mut().enumerate(){
            *ψz_val = Complex::<T>::new(
                (T::from_f64(-0.5).unwrap() * ((z[i] - mean[2]) /std[2]).powf(T::from_f64(2.0).unwrap())).exp(),
                T::zero(),
            );
        }
        let z_dims = Dim4::new(&[1, 1, parameters.size as u64, 1]);
        ψz = Array::new(&ψz_values, z_dims);
        normalize::<T>(&mut ψz, parameters.dx, parameters.dims);
        debug_assert!(check_norm::<T>(&ψz, parameters.dx, parameters.dims));
    } else {
        let z_dims = Dim4::new(&[1, 1, 1, 1]);
        ψz = Array::new(&[Complex::<T>::new(T::one(), T::zero())], z_dims);
    }
    


    // Construct ψ
    let ψ = mul(&ψx, &ψy, true);
    let mut ψ = mul(& ψ, &ψz, true);
    normalize::<T>(&mut ψ, parameters.dx, parameters.dims);
    debug_assert!(check_norm::<T>(&ψ, parameters.dx, parameters.dims));

    let ψk = crate::utils::fft::forward::<T>(&ψ, parameters.dims, parameters.size).unwrap();
    debug_assert!(check_norm::<T>(&ψk, parameters.dk, parameters.dims));
    
    SimulationObject::<T>::new_with_parameters(
        ψ,
        parameters.clone()
    )
}

pub fn cold_gauss_kspace<T>(
    mean: Vec<T>,
    std: Vec<T>,
    parameters: &SimulationParameters<T>,
    seed: Option<u64>,
) -> SimulationObject<T>
where
    T: Float + FloatingPoint + FromPrimitive + Display + Fromf64 + ConstGenerator<OutType=T> + HasAfEnum<AggregateOutType = T> + HasAfEnum<InType = T> + HasAfEnum<AbsOutType = T> + HasAfEnum<BaseType = T> + Fromf64 + WritableElement + ReadableElement + std::fmt::LowerExp + FloatConst,
    Complex<T>: HasAfEnum + ComplexFloating + FloatingPoint + HasAfEnum<ComplexOutType = Complex<T>> + HasAfEnum<UnaryOutType = Complex<T>> + HasAfEnum<AggregateOutType = Complex<T>> + HasAfEnum<AbsOutType = T>  + HasAfEnum<BaseType = T> + HasAfEnum<ArgOutType = T> + ConstGenerator<OutType=Complex<T>>,
    rand_distr::Standard: Distribution<T>,
{

    assert_eq!(mean.len(), parameters.dims as usize, "Cold Gauss k-Space: Mean vector provided has incorrect dimensionality");
    assert_eq!(std.len(), parameters.dims as usize, "Cold Gauss k-Space: Std vector provided has incorrect dimensionality");

    // Construct kspace grid
    let kx = get_kgrid::<T>(parameters.dx, parameters.size).to_vec();
    let ky = &kx;
    let kz = &kx;

    // Construct ψx
    let mut ψx_values = vec![Complex::<T>::new(T::zero(), T::zero()); parameters.size];
    for (i, ψx_val) in ψx_values.iter_mut().enumerate(){
        *ψx_val = Complex::<T>::new(
            (T::from_f64(-0.5).unwrap() * ((kx[i] - mean[0]) / std[0]).powf(T::from_f64(2.0).unwrap())).exp(),
            T::zero(),
        );
    }
    let x_dims = Dim4::new(&[parameters.size as u64, 1, 1, 1]);
    let mut ψx: Array<Complex<T>> = Array::new(&ψx_values, x_dims);
    normalize::<T>(&mut ψx, parameters.dk, parameters.dims);
    debug_assert!(check_norm::<T>(&ψx, parameters.dk, parameters.dims));

    // Construct ψy
    let mut ψy;
    if parameters.dims as usize >= 2 {
        let mut ψy_values = vec![Complex::<T>::new(T::zero(), T::zero()); parameters.size];
        for (i, ψy_val) in ψy_values.iter_mut().enumerate(){
            *ψy_val = Complex::<T>::new(
                (T::from_f64(-0.5).unwrap() * ((ky[i] - mean[1]) / std[1]).powf(T::from_f64(2.0).unwrap())).exp(),
                T::zero(),
            );
        }
        let y_dims = Dim4::new(&[1, parameters.size as u64, 1, 1]);
        ψy = Array::new(&ψy_values, y_dims);
        normalize::<T>(&mut ψy, parameters.dk, parameters.dims);
        debug_assert!(check_norm::<T>(&ψy, parameters.dk, parameters.dims));
    } else {
        let y_dims = Dim4::new(&[1, 1, 1, 1]);
        ψy = Array::new(&[Complex::<T>::new(T::one(), T::zero())], y_dims);
    }


    // Construct ψz
    let mut ψz;
    if parameters.dims as usize == 3 {
        let mut ψz_values = vec![Complex::<T>::new(T::zero(), T::zero()); parameters.size];
        for (i, ψz_val) in ψz_values.iter_mut().enumerate(){
            *ψz_val = Complex::<T>::new(
                (T::from_f64(-0.5).unwrap() * ((kz[i] - mean[2]) /std[2]).powf(T::from_f64(2.0).unwrap())).exp(),
                T::zero(),
            );
        }
        let z_dims = Dim4::new(&[1, 1, parameters.size as u64, 1]);
        ψz = Array::new(&ψz_values, z_dims);
        normalize::<T>(&mut ψz, parameters.dk, parameters.dims);
        debug_assert!(check_norm::<T>(&ψz, parameters.dk, parameters.dims));
    } else {
        let z_dims = Dim4::new(&[1, 1, 1, 1]);
        ψz = Array::new(&[Complex::<T>::new(T::one(), T::zero())], z_dims);
    }


    // Construct ψ in k space by multiplying the x, y, z functions just constructed.
    let ψ = mul(&ψx, &ψy, true);
    let mut ψ = mul(&ψ, &ψz, true);
    normalize::<T>(&mut ψ, parameters.dk, parameters.dims);
    debug_assert!(check_norm::<T>(&ψ, parameters.dk, parameters.dims));

    // Multiply random phases and then fft to get spatial ψ
    let seed = Some(seed.unwrap_or(0));
    let engine = RandomEngine::new(arrayfire::RandomEngineType::PHILOX_4X32_10, seed);
    let ψ_dims = Dim4::new(&[parameters.size as u64, parameters.size as u64, parameters.size as u64, 1]);
    let mut ψ = mul(
        &ψ,
        &exp(
            &mul(
                &complex_constant(
                    Complex::<T>::new(T::zero(),T::from_f64(2.0 * std::f64::consts::PI).unwrap()),
                    (parameters.size as u64, parameters.size as u64, parameters.size as u64, 1)
                ),
                &random_uniform::<T>(ψ_dims, &engine).cast(),
                false
            )
        ),
        false
    );
    debug_assert!(check_norm::<T>(&ψ, parameters.dk, parameters.dims));
    forward_inplace::<T>(&mut ψ, parameters.dims, parameters.size).expect("failed k-space -> spatial fft in cold gaussian kspace ic initialization");
    //normalize::<T>(&mut ψ, parameters.dx, parameters.dims);
    debug_assert!(check_norm::<T>(&ψ, parameters.dx, parameters.dims));



    SimulationObject::<T>::new_with_parameters(
        ψ,
        parameters.clone()
    )
}

pub fn cold_gauss_kspace_sample<T>(
    mean: Vec<T>,
    std: Vec<T>,
    parameters: &SimulationParameters<T>,
    scheme: SamplingScheme,
    phase_seed: Option<u64>,
    sample_seed: Option<u64>,
) -> SimulationObject<T>
where
    T: Float + FloatingPoint + FromPrimitive + Display + Fromf64 + ConstGenerator<OutType=T> + HasAfEnum<AggregateOutType = T> + HasAfEnum<InType = T> + HasAfEnum<AbsOutType = T> + HasAfEnum<BaseType = T> + Fromf64 + WritableElement + ReadableElement + FloatConst + std::fmt::LowerExp,
    Complex<T>: HasAfEnum + ComplexFloating + FloatingPoint + HasAfEnum<ComplexOutType = Complex<T>> + HasAfEnum<UnaryOutType = Complex<T>> + HasAfEnum<AggregateOutType = Complex<T>> + HasAfEnum<AbsOutType = T>  + HasAfEnum<BaseType = T> + HasAfEnum<ArgOutType = T> + ConstGenerator<OutType=Complex<T>>,
    rand_distr::Standard: Distribution<T>
{
    let mut simulation_object = cold_gauss_kspace::<T>(mean, std, parameters, phase_seed);
    sample_quantum_perturbation::<T>(&mut simulation_object.grid, &simulation_object.parameters, scheme, sample_seed);
    simulation_object
}


/// This function takes in some input and returns it with some noise based on given `n` and sampling method.
pub fn sample_quantum_perturbation<T>(
    grid: &mut SimulationGrid<T>,
    parameters: &SimulationParameters<T>,
    scheme: SamplingScheme,
    seed: Option<u64>,
)
where 
    T: Display + Float + FloatingPoint + FromPrimitive + Display + Fromf64 + ConstGenerator<OutType=T> + HasAfEnum<AggregateOutType = T> + HasAfEnum<InType = T> + HasAfEnum<AbsOutType = T> + HasAfEnum<BaseType = T> + Fromf64 + WritableElement + ReadableElement + FloatConst + ToPrimitive + std::fmt::LowerExp,
    Complex<T>: HasAfEnum + ComplexFloating + FloatingPoint + HasAfEnum<ComplexOutType = Complex<T>> + HasAfEnum<UnaryOutType = Complex<T>> + HasAfEnum<AggregateOutType = Complex<T>> + HasAfEnum<AbsOutType = T>  + HasAfEnum<BaseType = T> + HasAfEnum<ArgOutType = T> + ConstGenerator<OutType=Complex<T>>,
    rand_distr::Standard: Distribution<f64>
{
    // Unpack required quantities from simulation parameters
    let n: f64 = parameters.total_mass / parameters.particle_mass;
    let sqrt_n: T = T::from_f64(n.sqrt()).unwrap();
    let dim4 = get_dim4(parameters.dims, parameters.size);
    let ψ = &mut grid.ψ;

    // Convert input field to expected count per cell
    // TODO: Perhaps optimize mem storage by reusing ψ 
    let ψ_count: Array<Complex<T>> = mul(
        ψ, 
        &Complex::<T>::new(parameters.dx.powf(T::from_usize(parameters.dims as usize).unwrap()).sqrt(), T::zero()),
        true
    );

    // RNG engine
    let seed = Some(seed.unwrap_or(0));
    let engine = RandomEngine::new(arrayfire::RandomEngineType::PHILOX_4X32_10, seed);

    match scheme {

        SamplingScheme::Poisson => {

            println!("Poisson Scheme");

            // Sample poisson, take sqrt, and divide by sqrt of n
            let mut rng = rand::thread_rng();
            let sqrt_poisson_sample: Array<T> = {

                // Host array of norm squared to be able to sample from poisson
                let norm_sq_array: Array<T> = abs(&mul(ψ, &conjg(ψ), false)).cast();
                let mut norm_sq = vec![T::zero(); parameters.size.pow(parameters.dims as u32)];
                norm_sq_array.host(&mut norm_sq);

                // Iterate through vector, mapping x --> Poisson(x).sample().sqrt()
                let sample: Vec<T> = norm_sq
                    .iter()
                    .map(|&x| { 

                        // Poisson parameter is (probability mass in cell) * (total number of particles)
                        let pois_param: f64 = (x.to_f64().unwrap() * parameters.dx.to_f64().unwrap().powf(parameters.dims as u8 as f64)) * n;
                        debug_assert!(pois_param.is_finite());

                        // Sample poisson
                        let pois: Poisson<f64> = Poisson::new(pois_param).unwrap();
                        let a = pois.sample(&mut rng);

                        // Take poisson sample, divide by n, and take sqrt
                        let result = T::from_f64((a/n).sqrt()).unwrap();
                        debug_assert!(result.is_finite());

                        result
                    })
                    .collect();

                // poisson_sample return value
                Array::new(&sample, dim4)
            };

            // Multiply by original phases
            let ψ_: Array<Complex<T>> = mul(&sqrt_poisson_sample.cast(), &exp(&mul(&arg(ψ).cast(), &Complex::<T>::new(T::zero(),T::one()), true)), false).cast();

            // Finally, move data into ψ after converting count -> density
            *ψ = div(&ψ_, &Complex::<T>::new(parameters.dx.powf(T::from_usize(parameters.dims as usize).unwrap()).sqrt(), T::zero()), true);
        },

        SamplingScheme::Wigner => {

            println!("Wigner Sampling Scheme");
            
            // Sample independent Gaussian pairs --> Complex
            // pseudocode: add normal() + i*normal()
            let mut samples: Array<Complex<T>> = add(
                &mul(
                    &arrayfire::random_normal::<T>(dim4, &engine).cast(),
                    &complex_constant(Complex::<T>::new(T::one(),T::zero()), (1,1,1,1)),
                    true,
                ),
                &mul(
                    &arrayfire::random_normal::<T>(dim4, &engine).cast(),
                    &complex_constant(Complex::<T>::new(T::zero(),T::one()), (1,1,1,1)),
                    true
                ),
                false
            );

            // Scale the samples
            samples = div(
                &samples, 
                &complex_constant(Complex::<T>::new(sqrt_n*T::from_f64(2.0).unwrap(), T::zero()), (1,1,1,1)),
                true
            );


            // Add them to ψ_count
            let ψ_ = add(&ψ_count, &samples, false);

            // Finally, move data into ψ
            *ψ = div(&ψ_, &Complex::<T>::new(parameters.dx.powf(T::from_usize(parameters.dims as usize).unwrap()).sqrt(), T::zero()), true);
        },

        SamplingScheme::Husimi => {

            println!("Husimi Sampling Scheme");

            // Sample independent Gaussian pairs --> Complex
            // pseudocode: add normal() + i*normal()
            let mut samples: Array<Complex<T>> = add(
                &mul(
                    &arrayfire::random_normal::<T>(dim4, &engine).cast(),
                    &complex_constant(Complex::<T>::new(T::one(),T::zero()), (1,1,1,1)),
                    true,
                ),
                &mul(
                    &arrayfire::random_normal::<T>(dim4, &engine).cast(),
                    &complex_constant(Complex::<T>::new(T::zero(),T::one()), (1,1,1,1)),
                    true
                ),
                false
            );

            // Scale the samples
            samples = div(
                &samples, 
                &complex_constant(Complex::<T>::new(sqrt_n*T::from_f64(2.0).unwrap().sqrt(), T::zero()), (1,1,1,1)),
                true
            );

            // Add them to ψ_count
            let ψ_ = add(&ψ_count, &samples, false);

            // Finally, move data into ψ after converting count -> density
            *ψ = div(&ψ_, &Complex::<T>::new(parameters.dx.powf(T::from_usize(parameters.dims as usize).unwrap()).sqrt(), T::zero()), true);
        }
    }
}


pub fn user_specified_ics<T>(
    path: String,
) -> Array<Complex<T>> 
where 
    T: Float + FloatingPoint + FromPrimitive + Display + Fromf64 + ConstGenerator<OutType=T> + HasAfEnum<AggregateOutType = T> + HasAfEnum<InType = T> + HasAfEnum<AbsOutType = T> + HasAfEnum<BaseType = T> + Fromf64 + ndarray_npy::WritableElement + ndarray_npy::ReadableElement + std::fmt::LowerExp ,
    Complex<T>: HasAfEnum + ComplexFloating + FloatingPoint + HasAfEnum<ComplexOutType = Complex<T>> + HasAfEnum<UnaryOutType = Complex<T>> + HasAfEnum<AggregateOutType = Complex<T>> + HasAfEnum<AbsOutType = T>  + HasAfEnum<BaseType = T>,
{
    use ndarray::{Array1, Array2, Array3, ArrayBase};
    use ndarray_npy::NpzReader;
    use std::fs::File;

    // Open npz file
    let mut npz = NpzReader::new(File::open(path).expect("ics file does not exist")).expect("failed to read file as npz");

    // Read contents of file
    println!("{:?}", npz.names());
    let np_real: ArrayBase<OwnedRepr<T>, ndarray::IxDyn> = npz.by_name("real.npy").expect("couldn't read real part of field");
    let np_imag: ArrayBase<OwnedRepr<T>, ndarray::IxDyn> = npz.by_name("imag.npy").expect("couldn't read imag part of field");
    let dims: Dimensions = num::FromPrimitive::from_usize(np_real.ndim()).expect("User specified ICs have invalid number of dimensions.");
    let shape = np_real.shape();
    assert!({
            let mut check = true;
            let shape_1 = shape[0];
            for dim in 1..dims as usize {
                check = check && shape[dim] == shape_1;
            }
            check
        },
        "Only uniform grids are supported at this time"
    );
    let size = shape[0];

    // Turn into raw vectors
    let np_real: Vec<T> = np_real.into_raw_vec();
    let np_imag: Vec<T> = np_imag.into_raw_vec();
    println!("np_real is {}", np_real.len());


    // Construct complex data array
    let mut data: Vec<Complex<T>> = Vec::<Complex<T>>::with_capacity(size.pow(dims as u32));
    for (i, (&real, imag)) in np_real.iter().zip(np_imag).enumerate() {
        data.push(Complex::<T>::new(real, imag));
    }
    let dim4 = get_dim4(dims, size);
    let data: Array<Complex<T>> = Array::new(&data, dim4);
    
    // Return data
    data
}


fn get_dim4(dims: Dimensions, size: usize) -> Dim4 {
    match dims {
        Dimensions::One => Dim4::new(&[size as u64, 1, 1, 1]),
        Dimensions::Two => Dim4::new(&[size as u64, size as u64, 1, 1]),
        Dimensions::Three => Dim4::new(&[size as u64, size as u64, size as u64, 1]),
    }
}

#[derive(Serialize, Deserialize)]
pub enum SamplingScheme {
    Poisson,
    Wigner,
    Husimi,
}

#[test]
fn test_cold_gauss_initialization() {
    
    use arrayfire::{sum_all, conjg};
    use approx::assert_abs_diff_eq;

    // Gaussian parameters
    let mean = vec![0.5; 3];
    let std = vec![0.2; 3];

    type T = f32;

    // Simulation parameters
    const K: usize = 3;
    const S: usize = 512;
    let axis_length = 1.0;
    let time = 1.0;
    let total_sim_time = 1.0;
    let cfl = 0.25;
    let num_data_dumps = 100;
    let total_mass = 1.0;
    let particle_mass = 1.0;
    let sim_name = "cold-gauss".to_string();
    let k2_cutoff = 0.95;
    let alias_threshold = 0.02;
    let hbar_ = None;

    let parameters = SimulationParameters::<T>::new(
        axis_length,
        time,
        total_sim_time,
        cfl,
        num_data_dumps,
        total_mass,
        particle_mass,
        sim_name,
        k2_cutoff,
        alias_threshold,
        hbar_,
        num::FromPrimitive::from_usize(K).unwrap(),
        S
    );

    // Create a Simulation Object using Gaussian parameters and
    // simulation parameters 
    let sim: SimulationObject<T> = cold_gauss::<T>(
        mean,
        std,
        &parameters
    );

    let norm_check = sum_all(
        &mul(
            &sim.grid.ψ,
            &conjg(&sim.grid.ψ),
            false
        )
    ).0 * sim.parameters.dx.powf(K as T);

    //arrayfire::af_print!("ψ", slice(&sim.grid.ψ, S as i64 / 2));
    assert_abs_diff_eq!(
        norm_check,
        1.0,
        epsilon = 1e-6
    );
    assert!(check_norm::<T>(&sim.grid.ψ, sim.parameters.dx, num::FromPrimitive::from_usize(K).unwrap()));
}