use msm_data_interpreter::analyze_sims;
use glob::glob;
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

    let args: Vec<String> = env::args().collect();
    let sim_base_name: String = args[1].clone();
    let dump: usize = args[2].parse().unwrap();

    // let args = Args::parse();

    // let sim_base_name = "../rust/sim_data/bose-star-512".to_string();
    // let ndumps = 200;

    for sim in glob(format!("{}-stream*", sim_base_name).as_str()).unwrap() {
        println!("{}", sim.unwrap().display());
    }

    type T = f64;
    const K: usize = 3;
    const S: usize = 512;

    analyze_sims::<T, K, S>(sim_base_name, dump);
}