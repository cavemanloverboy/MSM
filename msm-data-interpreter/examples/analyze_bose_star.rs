use msm_data_interpreter::{analyze_sims, post_combine, balancer::Balancer, Functions, PostCombineFunctions};
use glob::glob;
use ndrustfft::{Complex, FftHandler, ndfft};
use ndarray::{Array4, ArrayBase, Dim, OwnedRepr};
use std::sync::Arc;
use std::env;

#[macro_use]
extern crate lazy_static;

fn main() {

    // type T = f64;
    const K: usize = 3;
    const S: usize = 512;
    

    // Define work scope and sim parameters
    let args: Vec<String> = env::args().collect();
    let sim_base_name: String = args[1].clone();
    let dumps: Vec<u16> = (0..=100_u16).collect::<Vec<u16>>();
    let axis_length: f64 = 30.0;
    lazy_static! {
      pub static ref DVOLUME: f64 = (30.0 / S as f64).powi(K as i32);
    }

    // Define Array4 -> Array4 functions that are applied to every stream and are reduced to a global state
    let array_functions: Vec<(&str, Box<(dyn Fn(&Array4<Complex<f64>>, &Array4<Complex<f64>>) -> ArrayBase<OwnedRepr<Complex<f64>>, Dim<[usize; 4]>> + Send + Sync + 'static)>)> = vec![
      ("psi", Box::new(|ψ: &Array4<Complex<f64>>, ψk: &Array4<Complex<f64>>| ψ.clone())),
      ("psi2", Box::new(|ψ: &Array4<Complex<f64>>, ψk: &Array4<Complex<f64>>| {
        ψ.map(|x| x*x.conj())
      })),
      ("psik", Box::new(|ψ: &Array4<Complex<f64>>, ψk: &Array4<Complex<f64>>| {
          ψk.clone()
        })),
      ("psik2", Box::new(|ψ: &Array4<Complex<f64>>, ψk: &Array4<Complex<f64>>| {
          ψk.map(|x| x*x.conj())
        }))
    ];

    // Define Array4 --> Complex functions that are applied to every stream and are reduce to a global state
    let scalar_functions:  Vec<(&str, Box<(dyn Fn(&Array4<Complex<f64>>, &Array4<Complex<f64>>) -> Complex<f64> + Send + Sync + 'static)>)> = vec![
      // ("Qk".to_string(), Box::new(|psi: &Array4<Complex<f64>>| {
      //   let mut fft_handler: ndrustfft::FftHandler<f64> = ndrustfft::FftHandler::new(S);
      //   let mut psi_buffer = psi.clone();
      //   let mut psik = psi.clone();
      //   for dim in 0..K {
      //       if dim > 0 { psik = psi_buffer.clone(); }
      //       ndfft(&psi, &mut psik, &mut fft_handler, dim);
      //       psi_buffer = psik.clone();
      //   }
      // }))
    ];
    let functions = Arc::new(Functions::<f64, _, K, S>::new(
      array_functions,
      scalar_functions,
    ));

    // Initialize mpi
    let (universe, threading) = mpi::initialize_with_threading(mpi::Threading::Funneled).expect("Failed to initialize mpi");
    let world = universe.world();
    let mut balancer = Balancer::<()>::new_from_world(world, 2);

    analyze_sims::<f64, K, S>(functions, sim_base_name, &dumps, &mut balancer);

    let post_array_functions: Vec<(&str, Box<(dyn Fn(&ArrayBase<OwnedRepr<Complex<f64>>, Dim<[usize; 4]>>, &ArrayBase<OwnedRepr<Complex<f64>>, Dim<[usize; 4]>>, &ArrayBase<OwnedRepr<Complex<f64>>, Dim<[usize; 4]>>, &ArrayBase<OwnedRepr<Complex<f64>>, Dim<[usize; 4]>>) -> Array4<Complex<f64>> + Send + Sync + 'static)>)> = vec![];
    let post_scalar_functions: Vec<(&str, Box<(dyn Fn(u16, &ArrayBase<OwnedRepr<Complex<f64>>, Dim<[usize; 4]>>, &ArrayBase<OwnedRepr<Complex<f64>>, Dim<[usize; 4]>>, &ArrayBase<OwnedRepr<Complex<f64>>, Dim<[usize; 4]>>, &ArrayBase<OwnedRepr<Complex<f64>>, Dim<[usize; 4]>>) -> (u16, Complex<f64>) + Send + Sync + 'static)>)> = vec![
      ("Qx", Box::new(|dump: u16, ψ: &Array4<Complex<f64>>, ψ2: &Array4<Complex<f64>>, ψk: &Array4<Complex<f64>>, ψk2: &Array4<Complex<f64>>| {
        (dump, (ψ2 - ψ.map(|x| x*x.conj())).sum() * *DVOLUME)
      })),
      ("Qk", Box::new(|dump: u16, ψ: &Array4<Complex<f64>>, ψ2: &Array4<Complex<f64>>, ψk: &Array4<Complex<f64>>, ψk2: &Array4<Complex<f64>>| {
        (dump, (ψk2 - ψk.map(|x| x*x.conj())).sum() * *DVOLUME)
      }))
    ];
    let post_combine_functions = Arc::new(PostCombineFunctions::new(
      post_array_functions,
      post_scalar_functions,
    ));

    let combined_base_name = format!("{}-combined/PLACEHOLDER_DUMP", sim_base_name);
    post_combine::<f64, _, K, S>(post_combine_functions, combined_base_name, &dumps, &mut balancer);
}
