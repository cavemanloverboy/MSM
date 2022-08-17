use arrayfire::{Dim4, Array, HasAfEnum};
use num::{Float, Complex};
use ndarray_npy::{WritableElement, write_npy};
use anyhow::Result;
use serde::de::DeserializeOwned;
use std::thread::{spawn, JoinHandle};
use std::sync::{Arc, RwLock};
use std::time::Instant;

/// This function writes an arrayfire array to disk in .npy format. It first hosts the
pub fn complex_array_to_disk<T>(
    path: String,
    array: &Array<Complex<T>>,
    shape: [u64; 4],
) -> Result<Vec<JoinHandle<Instant>>>
where
    T: Float + HasAfEnum + WritableElement + Send + 'static + Sync,
    Complex<T>: HasAfEnum,
{
    let timer = Instant::now();

    // Host array
    let mut host = vec![Complex::<T>::new(T::zero(), T::zero()); array.elements()];
    array.host(&mut host);

    // Atomic reference counters
    let real_host = Arc::new(RwLock::new(host));
    let imag_host = real_host.clone();

    // Construct path
    let real_path = format!("{}_real", path);
    let imag_path = format!("{}_imag", path);

    // Spawn a thread for each of the i/o operations
    let real_handle: std::thread::JoinHandle<_> = spawn(move || {

        // Gather real values
        let real: Vec<T> = real_host
            .read()
            .unwrap()
            .iter()
            .map(|x| x.re)
            .collect();

        // Build nd_array for npy serialization
        let real: ndarray::Array1<T> = ndarray::ArrayBase::from_vec(real);

        // Reshape 1D into 4D
        let real = real.into_shape(array_to_tuple(shape)).unwrap();

        array_to_disk(real_path, &real).expect("write to disk in parallel failed");

        timer
    });
    let imag_handle: std::thread::JoinHandle<_> = spawn(move || {

        // Gather imag values
        let imag: Vec<T> = imag_host
            .read()
            .unwrap()
            .iter()
            .map(|x| x.im)
            .collect();

        // Build nd_array for npy serialization
        let imag: ndarray::Array1<T> = ndarray::ArrayBase::from_vec(imag);

         // Reshape 1D into 4D
        let imag = imag.into_shape(array_to_tuple(shape)).unwrap();

        array_to_disk(imag_path, &imag).expect("write to disk in parallel failed");

        timer
    });

    Ok(vec![real_handle, imag_handle])


    // // Serial Async Scope
    // {
    //     // Host array
    //     let mut host = vec![Complex::<T>::new(T::zero(), T::zero()); array.elements()];
    //     array.host(&mut host);
    //     let real: Vec<T> = host
    //         .iter()
    //         .map(|x| x.re)
    //         .collect();
    //     let imag: Vec<T> = host
    //         .iter()
    //         .map(|x| x.im)
    //         .collect();
        
        
    //     // Build nd_array for npy serialization
    //     let real: ndarray::Array1<T> = ndarray::ArrayBase::from_vec(real);
    //     let imag: ndarray::Array1<T> = ndarray::ArrayBase::from_vec(imag);
    //     let real = real.into_shape(array_to_tuple(shape)).unwrap();
    //     let imag = imag.into_shape(array_to_tuple(shape)).unwrap();
    //     //println!("host shape is now {:?}", real.shape());

    //         // Write to npz
    //     //  let mut npz = NpzWriter::new(File::create(path).unwrap());
    //     //  npz.add_array("real", &real).context(RuntimeError::IOError)?;
    //     //  npz.add_array("imag", &imag).context(RuntimeError::IOError)?;
    //     //  npz.finish().context(RuntimeError::IOError)?;

    //     let real_path = format!("{}_real", path);
    //     let imag_path = format!("{}_imag", path);
    //     let real = array_to_disk(real_path, &real);
    //     let imag = array_to_disk(imag_path, &imag);
    //     futures::join!(real, imag);//.expect("write to disk in parallel failed");
    // }

    // println!("I/O took {} millis", timer.elapsed().as_millis());
    // Ok(())
}

fn array_to_disk<T>(
    path: String,
    array: &ndarray::Array4<T>,
) -> Result<()>
where
    T: Float + HasAfEnum + WritableElement,
{
     // Write to npy
     write_npy(path, array).expect("write to disk failed");
     Ok(())
}

/// This is a helper function to turn a length 4 array (required by Dim4) into a tuple,
/// which is required by ndarray::Array's .reshape() method
pub fn array_to_tuple(
    dim4: [u64; 4],
) -> (usize, usize, usize, usize) {
    (dim4[0] as usize, dim4[1] as usize, dim4[2] as usize, dim4[3] as usize)
}

/// This is a helper function to turn a length 4 array (required by Dim4) into a Dim4,
pub fn array_to_dim4(
    dim4: [u64; 4],
) -> Dim4 {
    Dim4::new(&dim4)
}

/// This function reads toml files
pub fn read_toml<T: DeserializeOwned>(
    path: String
) -> T {

    // Read toml config file
    let toml_contents: &str = &std::fs::read_to_string(&path).expect(format!("Unable to load toml and parse as string: {}", &path).as_str());

    // Return parsed toml from str
    toml::from_str(toml_contents).expect("Could not parse toml file contents")
}

use crate::ics::InitialConditions;
#[cfg(feature = "expanding")]
use crate::expanding::CosmologyParameters;


#[derive(Deserialize)]
pub struct TomlParameters {
    
    /// Physical length of box
    pub axis_length: f64,
    /// Start (and current) time of simulation
    pub time: Option<f64>,
    /// End time of simulation
    pub final_sim_time: f64,
    /// Safety factor
    pub cfl: f64,
    /// Number of data dumps
    pub num_data_dumps: u32,
    /// Total mass in the system
    pub total_mass: f64,
    /// Particle mass (use this or hbar_ = HBAR / particle_mass)
    pub particle_mass: Option<f64>,
    /// Specify number of particles. If this is used, `particle_mass` is ignored and HBAR const is ignored.
    pub ntot: Option<f64>,
    /// HBAR / particle mass (use this or particle_mass)
    pub hbar_: Option<f64>,
    /// Name of simulation (used for directories)
    pub sim_name: String,
    /// Where to begin checking for cutoff (this is a parameter in [0,1])
    pub k2_cutoff: f64,
    /// How much of mass to use as threshold for constitutes aliasing (P(k > k2_cutoff * k2))
    pub alias_threshold: f64,
    /// Dimensionality of grid
    pub dims: usize,
    /// Number of grid cells per dim
    pub size: usize,
    /// Initial Conditions
    pub ics: InitialConditions,
    
    /// Cosmological Parameters
    #[cfg(feature = "expanding")]
    pub cosmology: CosmologyParameters,

}