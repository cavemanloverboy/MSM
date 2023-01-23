use arrayfire::{device_info, set_device};
use clap::Parser;
use msm_common::{constants::*, read_toml, TomlParameters};
use msm_simulator::{simulation_object::*, utils::io::parameters_from_toml};
use std::error::Error;
use std::time::Instant;

#[derive(Parser)]
pub struct CommandLineArguments {
    #[clap(long, short)]
    toml: String,
    #[clap(long, short)]
    verbose: bool,
    #[clap(long)]
    test: bool,
}
fn main() -> Result<(), Box<dyn Error>> {
    // Set to gpu if available
    set_device(0);
    println!("{:?}", device_info());

    env_logger::builder()
        .format_timestamp_secs()
        // .filter_level(LevelFilter::Debug)
        .init();

    // Start timer
    let now = Instant::now();

    // Parse path to toml
    let args = CommandLineArguments::parse();

    // If seeds are being used, generate individual tomls for every stream
    let toml: TomlParameters = read_toml(&args.toml)?;
    let streams: Vec<SimulationParameters<_>> = parameters_from_toml(toml)?;
    let multi_stream: bool = streams.len() > 1;

    // Given a set of tomls, define simulation objects and run sims
    for stream in streams {
        // New sim obj from toml
        let mut simulation_object = SimulationObject::<f64>::new_from_params(stream)?;

        // Print simulation parameters and physical constants
        if args.verbose {
            println!();
            println!();
            println!(
                "Working on simulation {}",
                simulation_object.parameters.sim_name
            );
            println!("Simulation Parameters\n{}", simulation_object.parameters);
            println!("Physical Constants\nHBAR = {HBAR:.5e}\nPOIS_CONSTANT = {POIS_CONST:.5e}");
        }

        if !args.test {
            // Dump initial condition
            simulation_object.dump();

            // Main evolve loop
            let start = Instant::now();
            while simulation_object.not_finished() {
                simulation_object
                    .update(args.verbose)
                    .expect("failed to update");
            }

            if args.verbose {
                println!(
                    "Finished {} in {} seconds",
                    simulation_object.parameters.sim_name,
                    start.elapsed().as_secs()
                );
            }
        }
    }

    if multi_stream {
        println!(
            "Finished all streams in {} seconds",
            now.elapsed().as_secs()
        );
    }

    Ok(())
}
