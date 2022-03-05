pub mod config;
pub mod constants;
pub mod simulation;
pub mod utils;

use config::*;
use constants::*;
use simulation::*;

use ndarray::{array, Array};

//pub fn set_up_simulation()
pub fn run_simulation(args: Vec<String>) {
    let my_array: Array<f64, _> = array![1.0, 2.0, 3.0];
    let sim = simulation::Simulation {
        ψ: my_array.clone(),
        k_grid: my_array.clone(),
    };

    println!("{}", sim.ψ);
}
