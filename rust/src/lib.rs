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

    // Set size
    const K: usize = 3;
    const S: usize = 128;
    type T = f32;

    // Gaussian Parameters
    let mean: [T; 3] = [0.5; 3];
    let std: [T; 3] = [0.2; 3];

    // Simulation Parameters
    let n_grid: u32 = S as u32;
    let axis_length: T = 1.0; 
    let time: T = 0.0;
    let total_sim_time: T = 1.0;
    let num_data_dumps: u32 = 100;
    let total_mass: T = 1.0;
    let particle_mass: T = 1e-6;
    let sim_name: &'static str = "cold-gauss";
    let params = SimulationParameters::<T>::new(
        n_grid,
        axis_length,
        time,
        total_sim_time,
        num_data_dumps,
        total_mass,
        particle_mass,
        sim_name,
    );


    // Construct SimulationObject
    let simulation_object = ics::cold_gauss::<T, K, S>(
        mean,
        std,
        params,
    );

}