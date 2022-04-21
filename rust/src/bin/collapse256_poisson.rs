//use msm::run_simulation;
use arrayfire::{set_device, device_mem_info};
use msm::simulation_object::*;
use msm::ics;

fn main() {

     set_device(0);

     // Set size
     const K: usize = 3;
     const S: usize = 256;
     type T = f32;



     // Simulation Parameters
     let axis_length: T = 60.0; 
     let time: T = 0.0;
     let cfl: T = 0.25;
     let num_data_dumps: u32 = 200;
     let total_mass: f64 = 1e12 * 10.0;
     let particle_mass: f64 = (1e-22)*9.0e-67;
     let sim_name: &'static str = "cold-gauss";
     let total_sim_time: T = 10.0 / (msm::constants::POIS_CONST * total_mass / (axis_length as f64).powf(3.0)).sqrt() as T;
     //let total_sim_time: T = axis_length.powf(2.0) / (msm::constants::HBAR/particle_mass) as f32 / 1000.0;//200.0; //500000.0;
     let k2_cutoff = 0.95;
     let alias_threshold = 0.02;


     let params = SimulationParameters::<T, K, S>::new(
          axis_length,
          time,
          total_sim_time,
          cfl,
          num_data_dumps,
          total_mass,
          particle_mass,
          sim_name,
          k2_cutoff,
          alias_threshold
     );

     // Overconstrained parameters
     let kinetic_max = params.k2_max as f64/ 2.0;
     let kinetic_dt = 2.0 * std::f64::consts::PI / (kinetic_max * msm::constants::HBAR/particle_mass);
     let dτ = kinetic_dt*(msm::constants::HBAR/particle_mass)/axis_length.powf(2.0) as f64;
     let χ: f64 = total_mass*msm::constants::POIS_CONST*axis_length.powf(4.0 - K as T) as f64/(msm::constants::HBAR/particle_mass).powf(2.0);
     println!("dτ = {:e}, χ = {:e}, ΔT = {:e}", dτ, χ, total_sim_time);

     // // Simulation Parameters
     // let axis_length: T = 60.0; 
     // let time: T = 0.0;
     // let cfl: T = 0.25;
     // let num_data_dumps: u32 = 200;
     // let total_mass: f64 = 1e12;
     // let particle_mass: f64 = (1e-22)*9.0e-67;
     // let sim_name: &'static str = "cold-gauss";
     // let total_sim_time: T = 10.0/dτ as T;//200.0; //500000.0;
     // let k2_cutoff = 0.95;
     // let alias_threshold = 0.02;


     // let params = SimulationParameters::<T, K, S>::new(
     //      axis_length,
     //      time,
     //      total_sim_time,
     //      cfl,
     //      num_data_dumps,
     //      total_mass,
     //      particle_mass,
     //      sim_name,
     //      k2_cutoff,
     //      alias_threshold
     // );

     // // Overconstrained parameters
     // let kinetic_max = params.k2_max as f64/ 2.0;
     // let kinetic_dt = 2.0 * std::f64::consts::PI / (kinetic_max * msm::constants::HBAR/particle_mass);
     // let dτ = kinetic_dt*(msm::constants::HBAR/particle_mass)/axis_length.powf(2.0) as f64;
     // let χ: f64 = total_mass*msm::constants::POIS_CONST*axis_length.powf(3.0) as f64/(msm::constants::HBAR/particle_mass).powf(2.0);
     // println!("dτ = {:e}, χ = {:e}, ΔT = {:e}", dτ, χ, total_sim_time);
     

     // Gaussian Parameters
     let kmax = msm::utils::fft::get_kgrid::<T,S>(axis_length / S as f32)
          .iter()
          .fold(0.0, |acc, x| if x.abs() > acc {
               x.abs()
          } else { acc });
     println!("kmax is {}", kmax);
     let mean: [T; 3] = [0.0; 3];
     let std: [T; 3] = [kmax/10.0; 3];

     // Construct SimulationObject
     let mut simulation_object = {
               ics::cold_gauss_kspace_sample::<T, K, S>(mean, std, params, ics::SamplingScheme::Poisson)
     };
     simulation_object.dump();
     
     debug_assert!(msm::utils::grid::check_complex_for_nans(&simulation_object.grid.ψ));

     while simulation_object.not_finished() {
          simulation_object.update().expect("failed to update");
     }
     assert!(!!!simulation_object.not_finished())
}
