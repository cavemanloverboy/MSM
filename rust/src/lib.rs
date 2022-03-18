pub mod constants;
pub mod simulation_object;
pub mod utils;
pub mod ics;

// use config::*;
// use constants::*;
// use simulation::*;


//pub fn set_up_simulation()
pub fn run_simulation(_args: Vec<String>) {
   
    // parse args
    //

    // gen ics
    //let mut ψ = ics::new();

    // main loop
    loop {
        //ψ.update();

        
    }
}



#[test]
fn test_cold_gauss_sim_f32() {

    use crate::simulation_object::*;
    use crate::ics;
    use std::time::Instant;

    // Set size
    const K: usize = 3;
    const S: usize = 512;
    type T = f32;

    // Gaussian Parameters
    let mean: [T; 3] = [0.5; 3];
    let std: [T; 3] = [0.2; 3];

    // Simulation Parameters
    let axis_length: T = 1.0; 
    let time: T = 0.0;
    let total_sim_time: T = 1.0;
    let dt: T = 0.1;
    let num_data_dumps: u32 = 100;
    let total_mass: T = 1.0;
    let particle_mass: T = 1e-2;
    let sim_name: &'static str = "cold-gauss";
    let params = SimulationParameters::<T, S>::new(
        axis_length,
        time,
        total_sim_time,
        dt,
        num_data_dumps,
        total_mass,
        particle_mass,
        sim_name,
    );


    // Construct SimulationObject
    let mut simulation_object = ics::cold_gauss::<T, K, S>(
        mean,
        std,
        params,
    );

    {
        // use arrayfire::{save_image,slice, abs};
        // use crate::utils::grid::{check_norm, normalize};
        // use crate::utils::io::array_to_disk;
        // assert!(check_norm::<T, K>(&simulation_object.grid.ψ, simulation_object.parameters.dx));
        // let mut img_slice = slice(&simulation_object.grid.ψ, S as i64/2);
        // let slice_shape = [S as u64, S as u64, 1, 1];
        // array_to_disk("img_slice", "coldgauss", &abs(&img_slice), slice_shape);
    }

    // while simulation_object.not_finished() {
    //     simulation_object.update();
    // }
    // assert!(!!!simulation_object.not_finished())
}