use msm_data_interpreter::analyze_sims;
use glob::glob;
use msm_data_interpreter::Functions;
use ndrustfft::Complex;
use ndarray::Array4;
// use clap::Parser;

// /// Simple program to greet a person
// #[derive(Parser, Debug)]
// #[clap(author, version, about, long_about = None)]
// struct Args {
//     /// Name of the person to greet
//     #[clap(short, long)]
//     sim_base_name: String,

//     /// Number of times to greet
//     #[clap(short, long, default_value_t = 1)]
//     dump: usize,
// }
use std::env;

fn main() {

    type T = f64;
    const K: usize = 3;
    const S: usize = 512;
    

    let args: Vec<String> = env::args().collect();
    let sim_base_name: String = args[1].clone();
    let dump: usize = args[2].parse().unwrap();

    // let args = Args::parse();

    // let sim_base_name = "../rust/sim_data/bose-star-512".to_string();
    // let ndumps = 200;

    let array_functions = vec![
      ("psi", Box::new(|psi: &Array4<Complex<T>>| psi.clone()))
    ];
    let scalar_functions = vec![
      ("const", Box::new(|psi: &Array4<Complex<T>>| Complex::new(5.0 as T, 5.0)))
    ];
    let functions = Functions::<T, K, S>::new(
      array_functions,
      scalar_functions,
    );

    for sim in glob(format!("{}-stream*", sim_base_name).as_str()).unwrap() {
        println!("{}", sim.unwrap().display());
    }



    analyze_sims::<T, K, S>(functions, sim_base_name, &(0..=100_u16).collect::<Vec<u16>>());
}
