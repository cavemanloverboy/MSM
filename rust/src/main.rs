use arrayfire::{set_device,device_info};
use msm::simulation_object::*;
use msm::ics;
use std::time::Instant;
use std::convert::TryInto;
use clap::Parser;
use std::thread::sleep;
use msm::constants::*;

#[derive(Parser)]
pub struct CommandLineArguments {
    #[clap(long, short)]
    toml: String
}
fn main() {

    // Set to gpu if available
    set_device(0);
    println!("{:?}", device_info());

    // Start timer
    let now = Instant::now();

    // Parse path to toml
    let args = CommandLineArguments::parse();
    let toml_path = args.toml;
    
    // New sim obj from 
    let mut simulation_object = SimulationObject::<f64>::new_from_toml(toml_path);

    // Dump initial condition
    simulation_object.dump();

    // Print simulation parameters and physical constants
    println!("Simulation Parameters\n{}", simulation_object.parameters);
    println!("Physical Constants\nHBAR = {HBAR}\nPOIS_CONSTANT = {POIS_CONST}");
    sleep(std::time::Duration::from_secs(5));

    // Main evolve loop
    let verbose = true;
    while simulation_object.not_finished() {
        simulation_object.update(verbose).expect("failed to update");
    }

    println!("Finished simulation in {} seconds", now.elapsed().as_secs());
}
