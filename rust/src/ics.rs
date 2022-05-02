use crate::{
    simulation_object::*,
    utils::{
        grid::{normalize, check_norm},
        complex::complex_constant,
        fft::{forward_inplace, get_kgrid},
    },
};
use arrayfire::{Array, ComplexFloating, HasAfEnum, FloatingPoint, Dim4, add, mul, exp, random_uniform, conjg, arg, div, abs, Fromf64, ConstGenerator, RandomEngine, ImplicitPromote};
use num::{Complex, Float, FromPrimitive, ToPrimitive};
use num_traits::FloatConst;
use std::fmt::Display;
use std::iter::Iterator;
use rand_distr::{Poisson, Distribution};


/// This function produces initial conditions corresonding to a cold initial gaussian in sp
pub fn cold_gauss<T, const K: usize, const S: usize>(
    mean: [T; K],
    std: [T; K],
    params: SimulationParameters<T, K, S>,
) -> SimulationObject<T, K, S>
where
    T: Float + FloatingPoint + FromPrimitive + Display + Fromf64 + ConstGenerator<OutType=T> + HasAfEnum<AggregateOutType = T> + HasAfEnum<InType = T> + HasAfEnum<AbsOutType = T> + HasAfEnum<BaseType = T> + Fromf64 + ndarray_npy::WritableElement + std::fmt::LowerExp,
    Complex<T>: HasAfEnum + ComplexFloating + FloatingPoint + Default + HasAfEnum<ComplexOutType = Complex<T>> + HasAfEnum<UnaryOutType = Complex<T>> + HasAfEnum<AggregateOutType = Complex<T>> + HasAfEnum<AbsOutType = T>  + HasAfEnum<BaseType = T>,
{

    // Construct spatial grid
    let x: Vec<T> = (0..S)
        .map(|i| T::from_usize(i).unwrap() * params.dx)
        .collect();
    let y = &x;
    let z = &x;

    // Construct ψx
    let mut ψx_values = [Complex::<T>::new(T::zero(), T::zero()); S];
    for (i, ψx_val) in ψx_values.iter_mut().enumerate(){
        *ψx_val = Complex::<T>::new(
            (T::from_f64(-0.5).unwrap() * ((x[i] - mean[0]) / std[0]).powf(T::from_f64(2.0).unwrap())).exp(),
            T::zero(),
        );
    }
    let x_dims = Dim4::new(&[S as u64, 1, 1, 1]);
    let mut ψx: Array<Complex<T>> = Array::new(&ψx_values, x_dims);
    normalize::<T, K>(&mut ψx, params.dx);
    debug_assert!(check_norm::<T, K>(&ψx, params.dx));

    // Construct ψy
    let mut ψy;
    if K >= 2 {
        let mut ψy_values = [Complex::<T>::new(T::zero(), T::zero()); S];
        for (i, ψy_val) in ψy_values.iter_mut().enumerate(){
            *ψy_val = Complex::<T>::new(
                (T::from_f64(-0.5).unwrap() * ((y[i] - mean[1]) / std[1]).powf(T::from_f64(2.0).unwrap())).exp(),
                T::zero(),
            );
        }

        let y_dims = Dim4::new(&[1, S as u64, 1, 1]);
        ψy = Array::new(&ψy_values, y_dims);
        normalize::<T, K>(&mut ψy, params.dx);
        debug_assert!(check_norm::<T, K>(&ψy, params.dx));
    } else {
        let y_dims = Dim4::new(&[1, 1, 1, 1]);
        ψy = Array::new(&[Complex::<T>::new(T::one(), T::zero())], y_dims);
    }



    // Construct ψz
    let mut ψz;
    if K == 3 {
        let mut ψz_values = [Complex::<T>::new(T::zero(), T::zero()); S];
        for (i, ψz_val) in ψz_values.iter_mut().enumerate(){
            *ψz_val = Complex::<T>::new(
                (T::from_f64(-0.5).unwrap() * ((z[i] - mean[2]) /std[2]).powf(T::from_f64(2.0).unwrap())).exp(),
                T::zero(),
            );
        }
        let z_dims = Dim4::new(&[1, 1, S as u64, 1]);
        ψz = Array::new(&ψz_values, z_dims);
        normalize::<T, K>(&mut ψz, params.dx);
        debug_assert!(check_norm::<T, K>(&ψz, params.dx));
    } else {
        let z_dims = Dim4::new(&[1, 1, 1, 1]);
        ψz = Array::new(&[Complex::<T>::new(T::one(), T::zero())], z_dims);
    }
    


    // Construct ψ
    let ψ = mul(&ψx, &ψy, true);
    let mut ψ = mul(& ψ, &ψz, true);
    normalize::<T, K>(&mut ψ, params.dx);
    debug_assert!(check_norm::<T, K>(&ψ, params.dx));

    let ψk = crate::utils::fft::forward::<T, K, S>(&ψ).unwrap();
    debug_assert!(check_norm::<T, K>(&ψk, params.dk));
    
    SimulationObject::<T, K, S>::new(
        ψ,
        params.axis_length,
        params.time,
        params.total_sim_time,
        params.cfl,
        params.num_data_dumps,
        params.total_mass,
        params.particle_mass,
        params.sim_name,
        params.k2_cutoff,
        params.alias_threshold,
        Some(params.hbar_.to_f64().unwrap())
    )
}

pub fn cold_gauss_kspace<T, const K: usize, const S: usize>(
    mean: [T; K],
    std: [T; K],
    params: SimulationParameters<T, K, S>,
    seed: Option<u64>,
) -> SimulationObject<T, K, S>
where
    T: Float + FloatingPoint + FromPrimitive + Display + Fromf64 + ConstGenerator<OutType=T> + HasAfEnum<AggregateOutType = T> + HasAfEnum<InType = T> + HasAfEnum<AbsOutType = T> + HasAfEnum<BaseType = T> + Fromf64 + ndarray_npy::WritableElement + std::fmt::LowerExp,
    Complex<T>: HasAfEnum + ComplexFloating + FloatingPoint + Default + HasAfEnum<ComplexOutType = Complex<T>> + HasAfEnum<UnaryOutType = Complex<T>> + HasAfEnum<AggregateOutType = Complex<T>> + HasAfEnum<AbsOutType = T>  + HasAfEnum<BaseType = T>,
{

    // Construct kspace grid
    let kx = get_kgrid::<T, S>(params.dx).to_vec();
    let ky = &kx;
    let kz = &kx;

    // Construct ψx
    let mut ψx_values = [Complex::<T>::new(T::zero(), T::zero()); S];
    for (i, ψx_val) in ψx_values.iter_mut().enumerate(){
        *ψx_val = Complex::<T>::new(
            (T::from_f64(-0.5).unwrap() * ((kx[i] - mean[0]) / std[0]).powf(T::from_f64(2.0).unwrap())).exp(),
            T::zero(),
        );
    }
    let x_dims = Dim4::new(&[S as u64, 1, 1, 1]);
    let mut ψx: Array<Complex<T>> = Array::new(&ψx_values, x_dims);
    normalize::<T, K>(&mut ψx, params.dk);
    debug_assert!(check_norm::<T, K>(&ψx, params.dk));

    // Construct ψy
    let mut ψy;
    if K >= 2 {
        let mut ψy_values = [Complex::<T>::new(T::zero(), T::zero()); S];
        for (i, ψy_val) in ψy_values.iter_mut().enumerate(){
            *ψy_val = Complex::<T>::new(
                (T::from_f64(-0.5).unwrap() * ((ky[i] - mean[1]) / std[1]).powf(T::from_f64(2.0).unwrap())).exp(),
                T::zero(),
            );
        }
        let y_dims = Dim4::new(&[1, S as u64, 1, 1]);
        ψy = Array::new(&ψy_values, y_dims);
        normalize::<T, K>(&mut ψy, params.dk);
        debug_assert!(check_norm::<T, K>(&ψy, params.dk));
    } else {
        let y_dims = Dim4::new(&[1, 1, 1, 1]);
        ψy = Array::new(&[Complex::<T>::new(T::one(), T::zero())], y_dims);
    }


    // Construct ψz
    let mut ψz;
    if K == 3 {
        let mut ψz_values = [Complex::<T>::new(T::zero(), T::zero()); S];
        for (i, ψz_val) in ψz_values.iter_mut().enumerate(){
            *ψz_val = Complex::<T>::new(
                (T::from_f64(-0.5).unwrap() * ((kz[i] - mean[2]) /std[2]).powf(T::from_f64(2.0).unwrap())).exp(),
                T::zero(),
            );
        }
        let z_dims = Dim4::new(&[1, 1, S as u64, 1]);
        ψz = Array::new(&ψz_values, z_dims);
        normalize::<T, K>(&mut ψz, params.dk);
        debug_assert!(check_norm::<T, K>(&ψz, params.dk));
    } else {
        let z_dims = Dim4::new(&[1, 1, 1, 1]);
        ψz = Array::new(&[Complex::<T>::new(T::one(), T::zero())], z_dims);
    }


    // Construct ψ in k space by multiplying the x, y, z functions just constructed.
    let ψ = mul(&ψx, &ψy, true);
    let mut ψ = mul(&ψ, &ψz, true);
    normalize::<T, K>(&mut ψ, params.dk);
    debug_assert!(check_norm::<T, K>(&ψ, params.dk));

    // Multiply random phases and then fft to get spatial ψ
    let seed = Some(seed.unwrap_or(0));
    let engine = RandomEngine::new(arrayfire::RandomEngineType::PHILOX_4X32_10, seed);
    let ψ_dims = Dim4::new(&[S as u64, S as u64, S as u64, 1]);
    let mut ψ = mul(
        &ψ,
        &exp(
            &mul(
                &complex_constant(
                    Complex::<T>::new(T::zero(),T::from_f64(2.0 * std::f64::consts::PI).unwrap()),
                    (S as u64, S as u64, S as u64, 1)
                ),
                &random_uniform::<T>(ψ_dims, &engine).cast(),
                false
            )
        ),
        false
    );
    debug_assert!(check_norm::<T, K>(&ψ, params.dk));
    forward_inplace::<T, K, S>(&mut ψ).expect("failed k-space -> spatial fft in cold gaussian kspace ic initialization");
    //normalize::<T, K>(&mut ψ, params.dx);
    debug_assert!(check_norm::<T, K>(&ψ, params.dx));



    SimulationObject::<T, K, S>::new(
        ψ,
        params.axis_length,
        params.time,
        params.total_sim_time,
        params.cfl,
        params.num_data_dumps,
        params.total_mass,
        params.particle_mass,
        params.sim_name,
        params.k2_cutoff,
        params.alias_threshold,
        Some(params.hbar_.to_f64().unwrap())
    )
}

pub fn cold_gauss_kspace_sample<T, const K: usize, const S: usize>(
    mean: [T; K],
    std: [T; K],
    params: SimulationParameters<T, K, S>,
    scheme: SamplingScheme,
    phase_seed: Option<u64>,
    sample_seed: Option<u64>,
) -> SimulationObject<T, K, S>
where
    T: Float + FloatingPoint + FromPrimitive + Display + Fromf64 + ConstGenerator<OutType=T> + HasAfEnum<AggregateOutType = T> + HasAfEnum<InType = T> + HasAfEnum<AbsOutType = T> + HasAfEnum<BaseType = T> + Fromf64 + ndarray_npy::WritableElement + FloatConst + std::fmt::LowerExp,
    Complex<T>: HasAfEnum + ComplexFloating + FloatingPoint + Default + HasAfEnum<ComplexOutType = Complex<T>> + HasAfEnum<UnaryOutType = Complex<T>> + HasAfEnum<AggregateOutType = Complex<T>> + HasAfEnum<AbsOutType = T>  + HasAfEnum<BaseType = T> + HasAfEnum<ArgOutType = T> + ConstGenerator<OutType=Complex<T>>,
    rand_distr::Standard: Distribution<T>
{
    let mut simulation_object = cold_gauss_kspace::<T, K, S>(mean, std, params, phase_seed);
    sample_quantum_perturbation::<T, K, S>(&mut simulation_object.grid, &simulation_object.parameters, scheme, sample_seed);
    simulation_object
}


/// This function takes in some input and returns it with some noise based on given `n` and sampling method.
pub fn sample_quantum_perturbation<T, const K: usize, const S: usize>(
    grid: &mut SimulationGrid<T, K, S>,
    parameters: &SimulationParameters<T, K, S>,
    scheme: SamplingScheme,
    seed: Option<u64>,
)
where 
    T: Display + Float + FloatingPoint + FromPrimitive + Display + Fromf64 + ConstGenerator<OutType=T> + HasAfEnum<AggregateOutType = T> + HasAfEnum<InType = T> + HasAfEnum<AbsOutType = T> + HasAfEnum<BaseType = T> + Fromf64 + ndarray_npy::WritableElement + FloatConst + ToPrimitive + std::fmt::LowerExp,
    Complex<T>: HasAfEnum + ComplexFloating + FloatingPoint + Default + HasAfEnum<ComplexOutType = Complex<T>> + HasAfEnum<UnaryOutType = Complex<T>> + HasAfEnum<AggregateOutType = Complex<T>> + HasAfEnum<AbsOutType = T>  + HasAfEnum<BaseType = T> + HasAfEnum<ArgOutType = T> + HasAfEnum<ArgOutType = T> + ConstGenerator<OutType=Complex<T>>,
    rand_distr::Standard: Distribution<f64>
{
    // Unpack required quantities from simulation parameters
    let n: f64 = parameters.total_mass / parameters.particle_mass;
    let sqrt_n: T = T::from_f64(n.sqrt()).unwrap();
    let dims = get_dim4::<K, S>();
    let ψ = &mut grid.ψ;

    // Convert input field to expected count per cell
    // TODO: Perhaps optimize mem storage by reusing ψ 
    let ψ_count: Array<Complex<T>> = mul(
        ψ, 
        &Complex::<T>::new(parameters.dx.powf(T::from_usize(K).unwrap()).sqrt(), T::zero()),
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
                let mut norm_sq = vec![T::zero(); S.pow(K as u32)];
                norm_sq_array.host(&mut norm_sq);

                // Iterate through vector, mapping x --> Poisson(x).sample().sqrt()
                let sample: Vec<T> = norm_sq
                    .iter()
                    .map(|&x| { 

                        // Poisson parameter is (probability mass in cell) * (total number of particles)
                        let pois_param: f64 = (x.to_f64().unwrap() * parameters.dx.to_f64().unwrap().powf(K as f64)) * n;
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
                Array::new(&sample, dims)
            };

            // Multiply by original phases
            let ψ_: Array<Complex<T>> = mul(&sqrt_poisson_sample.cast(), &exp(&mul(&arg(ψ).cast(), &Complex::<T>::new(T::zero(),T::one()), true)), false).cast();

            // Finally, move data into ψ after converting count -> density
            *ψ = div(&ψ_, &Complex::<T>::new(parameters.dx.powf(T::from_usize(K).unwrap()).sqrt(), T::zero()), true);
        },

        SamplingScheme::Wigner => {

            println!("Wigner Sampling Scheme");
            
            // Sample independent Gaussian pairs --> Complex
            // pseudocode: add normal() + i*normal()
            let mut samples: Array<Complex<T>> = add(
                &mul(
                    &arrayfire::random_normal::<T>(dims, &engine).cast(),
                    &complex_constant(Complex::<T>::new(T::one(),T::zero()), (1,1,1,1)),
                    true,
                ),
                &mul(
                    &arrayfire::random_normal::<T>(dims, &engine).cast(),
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
            *ψ = div(&ψ_, &Complex::<T>::new(parameters.dx.powf(T::from_usize(K).unwrap()).sqrt(), T::zero()), true);
        },

        SamplingScheme::Husimi => {

            println!("Husimi Sampling Scheme");

            // Sample independent Gaussian pairs --> Complex
            // pseudocode: add normal() + i*normal()
            let mut samples: Array<Complex<T>> = add(
                &mul(
                    &arrayfire::random_normal::<T>(dims, &engine).cast(),
                    &complex_constant(Complex::<T>::new(T::one(),T::zero()), (1,1,1,1)),
                    true,
                ),
                &mul(
                    &arrayfire::random_normal::<T>(dims, &engine).cast(),
                    &complex_constant(Complex::<T>::new(T::zero(),T::one()), (1,1,1,1)),
                    true
                ),
                false
            );
            println!("var of samples.real is {:?}", arrayfire::var_all(&arrayfire::real(&samples), false));
            println!("var of samples.imag is {:?}", arrayfire::var_all(&arrayfire::imag(&samples), false));
            crate::utils::io::complex_array_to_disk("normals", "", &samples, [S as u64, S as u64, S as u64, 1]);


            // Scale the samples
            samples = div(
                &samples, 
                &complex_constant(Complex::<T>::new(sqrt_n*T::from_f64(2.0).unwrap().sqrt(), T::zero()), (1,1,1,1)),
                true
            );
            println!("n * var of div_samples.real is {}", arrayfire::var_all(&arrayfire::real(&samples), false).0*n);
            println!("n * var of div_samples.imag is {}", arrayfire::var_all(&arrayfire::imag(&samples), false).0*n);
            crate::utils::io::complex_array_to_disk("normals_divided", "", &samples, [S as u64, S as u64, S as u64, 1]);
            assert!(S == 1);

            // Add them to ψ_count
            let ψ_ = add(&ψ_count, &samples, false);

            // Finally, move data into ψ after converting count -> density
            *ψ = div(&ψ_, &Complex::<T>::new(parameters.dx.powf(T::from_usize(K).unwrap()).sqrt(), T::zero()), true);
        }
    }
}

fn get_dim4<const K: usize, const S: usize>() -> Dim4 {
    match K {
        1 => Dim4::new(&[S as u64, 1, 1, 1]),
        2 => Dim4::new(&[S as u64, S as u64, 1, 1]),
        3 => Dim4::new(&[S as u64, S as u64, S as u64, 1]),
        _ => panic!("Invalid Number of Dimensions")
    }
}

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
    let mean = [0.5; 3];
    let std = [0.2; 3];

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

    let params = SimulationParameters::<T, K, S>::new(
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
        hbar_
    );

    // Create a Simulation Object using Gaussian parameters and
    // simulation parameters 
    let sim: SimulationObject<T, K, S> = cold_gauss::<T, K, S>(
        mean,
        std,
        params
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
    assert!(check_norm::<T,K>(&sim.grid.ψ, sim.parameters.dx));
}