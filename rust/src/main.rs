use arrayfire::{set_device,device_info};
use msm::simulation_object::*;
use std::time::Instant;
use clap::Parser;
use std::thread::sleep;
use msm::constants::*;

#[derive(Parser)]
pub struct CommandLineArguments {
    #[clap(long, short)]
    toml: Vec<String>,
    #[clap(long, short)]
    verbose: bool,
}
fn main() {

    // Set to gpu if available
    set_device(0);
    println!("{:?}", device_info());

    // Start timer
    let now = Instant::now();

    // Parse path to tomls
    let args = CommandLineArguments::parse();
    
    // Given a set of tomls, define simulation objects and run sims
    for toml in args.toml.clone() {

        // New sim obj from toml
        let mut simulation_object = SimulationObject::<f64>::new_from_toml(toml);

        // Dump initial condition
        simulation_object.dump();

        // Print simulation parameters and physical constants
        println!("Working on simulation {}", simulation_object.parameters.sim_name);
        println!("Simulation Parameters\n{}", simulation_object.parameters);
        println!("Physical Constants\nHBAR = {HBAR}\nPOIS_CONSTANT = {POIS_CONST}");

        // Main evolve loop
        let start = Instant::now();
        while simulation_object.not_finished() {
            simulation_object.update(args.verbose).expect("failed to update");
        }
        println!("Finished {} in {} seconds", simulation_object.parameters.sim_name, start.elapsed().as_secs());
    }

    if args.toml.len() > 1 {
        println!("Finished simulations in {} seconds", now.elapsed().as_secs());
    }
}
