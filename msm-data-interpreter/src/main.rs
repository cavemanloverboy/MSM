use clap::Parser;
use msm::utils::io::TomlParameters;
use msm_data_interpreter::{analyze_sims, post_combine, Functions, PostCombineFunctions};
use ndarray::{Array4, ArrayBase, Dim, OwnedRepr};
use ndrustfft::Complex;
use std::env;
use std::error::Error;
use std::sync::Arc;

#[cfg(feature = "balancer")]
use msm_data_interpreter::balancer::Balancer;
#[cfg(not(feature = "balancer"))]
use msm_data_interpreter::balancer_nompi::Balancer;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// path to sim data (just base name, without stream-XXXXX)
    #[arg(short, long)]
    path: String,

    /// path to toml used to run simulations
    #[arg(short, long)]
    toml: String,
}

fn main() -> Result<(), Box<dyn Error>> {
    println!("Starting DI");

    let args = Args::parse();

    let toml: TomlParameters = msm::utils::io::read_toml(&args.toml)?;

    // Dimensionality and grid size
    let dims: usize = toml.dims;
    let size: usize = toml.size;

    // Define work scope and sim parameters
    let sim_base_name: String = args.path;
    let dumps: Vec<u16> = (0..=toml.num_data_dumps as u16).collect::<Vec<u16>>();
    let dv: f64 = (toml.axis_length / size as f64).powi(dims as i32);

    // Define Array4 -> Array4 functions that are applied to every stream and are reduced to a global state
    #[allow(unused)]
    let array_functions: Vec<(
        &str,
        Box<
            (dyn Fn(
                &Array4<Complex<f64>>,
                &Array4<Complex<f64>>,
            ) -> ArrayBase<OwnedRepr<Complex<f64>>, Dim<[usize; 4]>>
                 + Send
                 + Sync
                 + 'static),
        >,
    )> = vec![
        (
            "psi",
            Box::new(|ψ: &Array4<Complex<f64>>, ψk: &Array4<Complex<f64>>| ψ.clone()),
        ),
        (
            "psi2",
            Box::new(|ψ: &Array4<Complex<f64>>, ψk: &Array4<Complex<f64>>| ψ.map(|x| x * x.conj())),
        ),
        (
            "psik",
            Box::new(|ψ: &Array4<Complex<f64>>, ψk: &Array4<Complex<f64>>| ψk.clone()),
        ),
        (
            "psik2",
            Box::new(|ψ: &Array4<Complex<f64>>, ψk: &Array4<Complex<f64>>| {
                ψk.map(|x| x * x.conj())
            }),
        ),
    ];

    // Define Array4 --> Complex functions that are applied to every stream and are reduce to a global state
    let scalar_functions: Vec<(
        &str,
        Box<
            (dyn Fn(&Array4<Complex<f64>>, &Array4<Complex<f64>>) -> Complex<f64>
                 + Send
                 + Sync
                 + 'static),
        >,
    )> = vec![
        // leaving this here as an example
        // (
        //     "Qk",
        //     Box::new(|psi: &Array4<Complex<f64>>, psik: &Array4<Complex<f64>>| psi.sum()),
        // ),
    ];
    let functions = Arc::new(Functions::<f64, &str>::new(
        array_functions,
        scalar_functions,
        dims,
        size,
    ));

    // Initialize mpi
    #[cfg(feature = "balancer")]
    let mut balancer = {
        println!("initializing mpi");
        let (universe, threading) = mpi::initialize_with_threading(mpi::Threading::Funneled)
            .expect("Failed to initialize mpi");
        let world = universe.world();
        Balancer::<()>::new(world, 5)
    };
    #[cfg(not(feature = "balancer"))]
    let mut balancer = Balancer::<()>::new(5);

    println!("begin analyze_sims");
    analyze_sims::<f64, &str>(functions, &sim_base_name, &dumps, &mut balancer)?;

    let post_array_functions: Vec<(
        &str,
        Box<
            (dyn Fn(
                &ArrayBase<OwnedRepr<Complex<f64>>, Dim<[usize; 4]>>,
                &ArrayBase<OwnedRepr<Complex<f64>>, Dim<[usize; 4]>>,
                &ArrayBase<OwnedRepr<Complex<f64>>, Dim<[usize; 4]>>,
                &ArrayBase<OwnedRepr<Complex<f64>>, Dim<[usize; 4]>>,
            ) -> Array4<Complex<f64>>
                 + Send
                 + Sync
                 + 'static),
        >,
    )> = vec![];
    let post_scalar_functions: Vec<(
        &str,
        Box<
            (dyn Fn(
                u16,
                &ArrayBase<OwnedRepr<Complex<f64>>, Dim<[usize; 4]>>,
                &ArrayBase<OwnedRepr<Complex<f64>>, Dim<[usize; 4]>>,
                &ArrayBase<OwnedRepr<Complex<f64>>, Dim<[usize; 4]>>,
                &ArrayBase<OwnedRepr<Complex<f64>>, Dim<[usize; 4]>>,
            ) -> (u16, Complex<f64>)
                 + Send
                 + Sync
                 + 'static),
        >,
    )> = vec![(
        "Qx",
        Box::new(
            #[allow(unused)]
            move |dump: u16,
                  ψ: &Array4<Complex<f64>>,
                  ψ2: &Array4<Complex<f64>>,
                  ψk: &Array4<Complex<f64>>,
                  ψk2: &Array4<Complex<f64>>| {
                (dump, (ψ2 - ψ.map(|x| x * x.conj())).sum() * dv)
            },
        ),
    )];
    let post_combine_functions = Arc::new(PostCombineFunctions::new(
        post_array_functions,
        post_scalar_functions,
        dims,
        size,
    ));

    let combined_base_name = format!("{}-combined/PLACEHOLDER_DUMP", sim_base_name);
    post_combine::<f64, _>(
        post_combine_functions,
        combined_base_name,
        &dumps,
        &mut balancer,
    )?;

    Ok(())
}
