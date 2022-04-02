//use msm::run_simulation;
use arrayfire::{set_device, device_mem_info};
use msm::simulation_object::*;
use msm::ics;

fn main() {

     let order1 = false;
     let kspace_phase_shuffle = true; // need to change mean/std, as this is mean/std of spatial vs kspace

     if !order1{

          // Set size
          const K: usize = 3;
          const S: usize = 128;
          type T = f32;



          // Simulation Parameters
          let axis_length: T = 60.0; 
          let time: T = 0.0;
          let cfl: T = 0.25;
          let num_data_dumps: u32 = 200;
          let total_mass: f64 = 1e12;
          let particle_mass: f64 = (1e-22)*9.0e-67;
          let sim_name: &'static str = "cold-gauss";
          let total_sim_time: T = axis_length.powf(2.0) / (msm::constants::HBAR/particle_mass) as f32 / 1000.0;//200.0; //500000.0;
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
          let χ: f64 = total_mass*msm::constants::POIS_CONST*axis_length.powf(3.0) as f64/(msm::constants::HBAR/particle_mass).powf(2.0);
          println!("dτ = {:e}, χ = {:e}, ΔT = {:e}", dτ, χ, total_sim_time);
          

          // Gaussian Parameters
          let kmax = msm::utils::fft::get_kgrid::<T,S>(axis_length / S as f32)
               .iter()
               .fold(0.0, |acc, x| if x.abs() > acc {
                    x.abs()
               } else { acc });
          println!("kmax is {}", kmax);
          let mean: [T; 3] = [0.0; 3];
          let std: [T; 3] = [0.00*axis_length + kmax/5.0; 3];

          // Construct SimulationObject
          let mut simulation_object = {
               if !!!kspace_phase_shuffle {
                    ics::cold_gauss::<T, K, S>(mean, std, params)
               } else {
                    ics::cold_gauss_kspace::<T, K, S>(mean, std, params)
               }
          };
          simulation_object.dump();
          
          debug_assert!(msm::utils::grid::check_complex_for_nans(&simulation_object.grid.ψ));

          while simulation_object.not_finished() {
               simulation_object.update().expect("failed to update");
          }
          assert!(!!!simulation_object.not_finished())
     } else {
          set_device(0);
          println!("{:?}", device_mem_info());
     
          // Set size
          const K: usize = 3;
          const S: usize = 128;
          type T = f32;
     

     
          // Simulation Parameters
          let axis_length: T = 1.0; 
          let time: T = 0.0;
          let total_sim_time: T = 4.0;
          let cfl: T = 0.5;
          let num_data_dumps: u32 = 200;
          let total_mass: f64 = 1.0;
          let particle_mass: f64 = 1e-1;
          let sim_name: &'static str = "cold-gauss";
          let k2_cutoff = 0.95;
          let alias_threshold = 0.02;

          // Gaussian Parameters
          let kmax = msm::utils::fft::get_kgrid::<T,S>(axis_length / S as f32)
          .iter()
          .fold(0.0, |acc, x| if x.abs() > acc {
               x.abs()
          } else { acc });
          let mean: [T; 3] = [0.0; 3];
          let std: [T; 3] = [kmax/5.0; 3];


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
               alias_threshold,
          );

          // Overconstrained parameters
          let χ: f64 = total_mass*msm::constants::POIS_CONST*axis_length.powf(3.0) as f64/(msm::constants::HBAR/particle_mass).powf(2.0);
          println!("χ = {}", χ);
     
          // Construct SimulationObject
          let mut simulation_object = {
               if !!!kspace_phase_shuffle {
                    ics::cold_gauss::<T, K, S>(mean, std, params)
               } else {
                    ics::cold_gauss_kspace::<T, K, S>(mean, std, params)
               }
          };
          debug_assert!(msm::utils::grid::check_complex_for_nans(&simulation_object.grid.ψ));
     
          while simulation_object.not_finished() {
          simulation_object.update().expect("failed to update");
          //pb.inc(u64::from_f64().unwrap());
          }
          assert!(!!!simulation_object.not_finished())
     }
}
