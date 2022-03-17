use crate::{
    simulation_object::*,
    utils::grid::{normalize, check_norm},
};
use arrayfire::{Array, ComplexFloating, HasAfEnum, FloatingPoint, Dim4, mul, Fromf64, ConstGenerator};
use num::{Complex, Float, FromPrimitive};
use std::fmt::Display;
use std::iter::Iterator;


pub fn cold_gauss<T, const K: usize, const S: usize>(
    mean: [T; 3],
    std: [T; 3],
    params: SimulationParameters<T>,
) -> SimulationObject<T, K, S>
where
    T: Float + FloatingPoint + FromPrimitive + Display + Fromf64 + ConstGenerator<OutType=T> + HasAfEnum<InType = T> + HasAfEnum<BaseType = T> + Fromf64,
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
    let mut ψy_values = [Complex::<T>::new(T::zero(), T::zero()); S];
    for (i, ψy_val) in ψy_values.iter_mut().enumerate(){
        *ψy_val = Complex::<T>::new(
            (T::from_f64(-0.5).unwrap() * ((y[i] - mean[1]) / std[1]).powf(T::from_f64(2.0).unwrap())).exp(),
            T::zero(),
        );
    }
    let y_dims = Dim4::new(&[1, S as u64, 1, 1]);
    let mut ψy = Array::new(&ψy_values, y_dims);
    normalize::<T, K>(&mut ψy, params.dx);
    debug_assert!(check_norm::<T, K>(&ψy, params.dx));


    // Construct ψz
    let mut ψz_values = [Complex::<T>::new(T::zero(), T::zero()); S];
    for (i, ψz_val) in ψz_values.iter_mut().enumerate(){
        *ψz_val = Complex::<T>::new(
            (T::from_f64(-0.5).unwrap() * ((z[i] - mean[2]) /std[2]).powf(T::from_f64(2.0).unwrap())).exp(),
            T::zero(),
        );
    }
    let z_dims = Dim4::new(&[1, 1, S as u64, 1]);
    let mut ψz = Array::new(&ψz_values, z_dims);
    normalize::<T, K>(&mut ψz, params.dx);
    debug_assert!(check_norm::<T, K>(&ψz, params.dx));


    // Construct ψ
    let ψ = mul(&ψx, &ψy, true);
    let mut ψ = mul(& ψ, &ψz, true);
    normalize::<T, K>(&mut ψ, params.dx);
    debug_assert!(check_norm::<T, K>(&ψ, params.dx));
    SimulationObject::<T, K, S>::new(
        ψ,
        params.n_grid,
        params.axis_length,
        params.time,
        params.total_sim_time,
        params.dt,
        params.num_data_dumps,
        params.total_mass,
        params.particle_mass,
        params.sim_name,
    )
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
    const S: usize = 128;
    let n_grid = S as u32;
    let axis_length = 1.0;
    let time = 1.0;
    let total_sim_time = 1.0;
    let dt = 0.1;
    let num_data_dumps = 100;
    let total_mass = 1.0;
    let particle_mass = 1.0;
    let sim_name = "cold-gauss";
    let params = SimulationParameters::<T>::new(
        n_grid,
        axis_length,
        time,
        total_sim_time,
        dt,
        num_data_dumps,
        total_mass,
        particle_mass,
        sim_name,
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