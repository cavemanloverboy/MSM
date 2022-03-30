//use msm::run_simulation;
use arrayfire::{set_device, device_mem_info};
use msm::simulation_object::*;
use msm::ics;

fn main() {

     let order1 = false;

     if !order1{

          // Set size
          const K: usize = 3;
          const S: usize = 128;
          type T = f32;



          // Simulation Parameters
          let axis_length: T = 60.0; 
          let time: T = 0.0;
          let total_sim_time: T = 1000.0; //500000.0;
          let cfl: T = 0.05;
          let num_data_dumps: u32 = 200;
          let total_mass: f64 = 1e12;
          let particle_mass: f64 = (1e-22)*9.0e-67;
          let sim_name: &'static str = "cold-gauss";
          let params = SimulationParameters::<T, S>::new(
               axis_length,
               time,
               total_sim_time,
               cfl,
               num_data_dumps,
               total_mass,
               particle_mass,
               sim_name,
          );

          // Gaussian Parameters
          let mean: [T; 3] = [0.5*axis_length; 3];
          let std: [T; 3] = [0.2*axis_length; 3];

          // Construct SimulationObject
          let mut simulation_object = ics::cold_gauss::<T, K, S>(
               mean,
               std,
               params,
          );
          debug_assert!(msm::utils::grid::check_complex_for_nans(&simulation_object.grid.ψ));

          while simulation_object.not_finished() {
               simulation_object.update();
          }
          assert!(!!!simulation_object.not_finished())
     } else {
          set_device(0);
          println!("{:?}", device_mem_info());
     
          // Set size
          const K: usize = 3;
          const S: usize = 128;
          type T = f32;
     
          // Gaussian Parameters
          let mean: [T; 3] = [0.5; 3];
          let std: [T; 3] = [0.2; 3];
     
          // Simulation Parameters
          let axis_length: T = 1.0; 
          let time: T = 0.0;
          let total_sim_time: T = 1.0;
          let cfl: T = 0.05;
          let num_data_dumps: u32 = 200;
          let total_mass: f64 = 1.0;
          let particle_mass: f64 = 1e-1;
          let sim_name: &'static str = "cold-gauss";
          let params = SimulationParameters::<T, S>::new(
          axis_length,
          time,
          total_sim_time,
          cfl,
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
          debug_assert!(msm::utils::grid::check_complex_for_nans(&simulation_object.grid.ψ));
     
          while simulation_object.not_finished() {
          simulation_object.update();
          //pb.inc(u64::from_f64().unwrap());
          }
          assert!(!!!simulation_object.not_finished())
     }
}
