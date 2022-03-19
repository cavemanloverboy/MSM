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



// #[test]
// fn test_cold_gauss_sim_f32() {

//     use crate::simulation_object::*;
//     use crate::ics;
//     use std::time::Instant;

//     // Set size
//     const K: usize = 3;
//     const S: usize = 128;
//     type T = f32;

//     // Gaussian Parameters
//     let mean: [T; 3] = [0.5; 3];
//     let std: [T; 3] = [0.2; 3];

//     // Simulation Parameters
//     let axis_length: T = 60.0; 
//     let time: T = 0.0;
//     let total_sim_time: T = 500000.0;
//     let cfl: T = 0.05;
//     let num_data_dumps: u32 = 200;
//     let total_mass: T = 1e12;
//     let particle_mass: T = (1e-22)*9.0e-67;
//     let sim_name: &'static str = "cold-gauss";
//     let params = SimulationParameters::<T, S>::new(
//         axis_length,
//         time,
//         total_sim_time,
//         cfl,
//         num_data_dumps,
//         total_mass,
//         particle_mass,
//         sim_name,
//     );

//     // Construct SimulationObject
//     let mut simulation_object = ics::cold_gauss::<T, K, S>(
//         mean,
//         std,
//         params,
//     );
//     debug_assert!(crate::utils::grid::check_complex_for_nans(&simulation_object.grid.ψ));

//     use indicatif::ProgressBar;
//     let pb = ProgressBar::new(1000);
//     use num::FromPrimitive;
//     let mut tracker = 0;
//     while simulation_object.not_finished() {
//         simulation_object.update();
//         //pb.inc(u64::from_f64().unwrap());
//     }
//     assert!(!!!simulation_object.not_finished())
// }

// #[test]
// fn test_cold_gauss_sim_f32_order1() {

//     use arrayfire::{set_device, device_mem_info};
//     use crate::simulation_object::*;
//     use crate::ics;
//     use std::time::Instant;

//     set_device(0);
//     println!("{:?}", device_mem_info());

//     // Set size
//     const K: usize = 3;
//     const S: usize = 128;
//     type T = f32;

//     // Gaussian Parameters
//     let mean: [T; 3] = [0.5; 3];
//     let std: [T; 3] = [0.2; 3];

//     // Simulation Parameters
//     let axis_length: T = 1.0; 
//     let time: T = 0.0;
//     let total_sim_time: T = 1.0;
//     let cfl: T = 0.05;
//     let num_data_dumps: u32 = 200;
//     let total_mass: T = 1.0;
//     let particle_mass: T = 1e-1;
//     let sim_name: &'static str = "cold-gauss";
//     let params = SimulationParameters::<T, S>::new(
//         axis_length,
//         time,
//         total_sim_time,
//         cfl,
//         num_data_dumps,
//         total_mass,
//         particle_mass,
//         sim_name,
//     );

//     // Construct SimulationObject
//     let mut simulation_object = ics::cold_gauss::<T, K, S>(
//         mean,
//         std,
//         params,
//     );
//     debug_assert!(crate::utils::grid::check_complex_for_nans(&simulation_object.grid.ψ));

//     while simulation_object.not_finished() {
//         simulation_object.update();
//         //pb.inc(u64::from_f64().unwrap());
//     }
//     assert!(!!!simulation_object.not_finished())
// }