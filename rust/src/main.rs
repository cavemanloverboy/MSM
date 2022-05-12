use arrayfire::{set_device,device_info};
use msm::simulation_object::*;
use msm::ics;
use std::time::Instant;
use std::convert::TryInto;
use clap::Parser;

#[derive(Parser)]
pub struct CommandLineArguments {
    #[clap(long, short)]
    toml: String
}
fn main() {

    // Set to gpu if available
    set_device(0);
    device_info();

    // Start timer
    let now = Instant::now();

    // Parse path to toml
    let args = CommandLineArguments::parse();
    let toml_path = args.toml;
    
    // New sim obj from 
    let mut simulation_object = SimulationObject::<f64>::new_from_toml(toml_path);

    // Dump initial condition
    simulation_object.dump();
          
    // Main evolve loop
    let verbose = false;
    while simulation_object.not_finished() {
        simulation_object.update(verbose).expect("failed to update");
    }

    println!("Finished simulation in {} seconds", now.elapsed().as_secs());
}
