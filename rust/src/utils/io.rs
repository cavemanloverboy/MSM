use anyhow::Result;
use arrayfire::{Array, ConstGenerator, Dim4, FloatingPoint, Fromf64, HasAfEnum};
use ndarray_npy::{write_npy, WritableElement};
use num::{Complex, Float, FromPrimitive};
use regex::Regex;
use serde::de::DeserializeOwned;
use serde::Deserialize;
use serde_derive::Deserialize;
use std::fmt::Display;
use std::sync::Arc;
use std::thread::{spawn, JoinHandle};
use std::time::Instant;

use crate::constants::HBAR;
#[cfg(feature = "expanding")]
use crate::expanding::CosmologyParameters;
use crate::simulation_object::SimulationParameters;

use super::error::RuntimeError;
use crate::ics::{InitialConditions, SamplingParameters, SamplingScheme};

#[cfg(feature = "remote-storage")]
use {
    shadow_drive_sdk::{
        models::{storage_acct::StorageAcct, ShadowFile},
        ShadowDriveClient,
    },
    solana_sdk::{
        pubkey::Pubkey,
        signature::read_keypair_file,
        signer::{keypair::Keypair, Signer},
    },
    std::{path::PathBuf, sync::RwLock},
    tokio::runtime::{Builder, Runtime},
};

/// This function writes an arrayfire array to disk in .npy format. It first hosts the
pub fn complex_array_to_disk<T>(
    path: String,
    array: &Array<Complex<T>>,
    shape: [u64; 4],
) -> Result<Vec<JoinHandle<u128>>>
where
    T: Float + HasAfEnum + WritableElement + Send + 'static + Sync,
    Complex<T>: HasAfEnum,
{
    let timer = Instant::now();

    // Host array
    let mut host = vec![Complex::<T>::new(T::zero(), T::zero()); array.elements()];
    array.host(&mut host);

    // Atomic reference counters
    let real_host = Arc::new(host);
    let imag_host = real_host.clone();

    // Construct path
    let real_path = format!("{}_real", path);
    let imag_path = format!("{}_imag", path);

    // Spawn a thread for each of the i/o operations
    let real_handle: std::thread::JoinHandle<_> = spawn(move || {
        // Gather real values
        let real: Vec<T> = real_host.iter().map(|x| x.re).collect();

        // Build nd_array for npy serialization
        let real: ndarray::Array1<T> = ndarray::ArrayBase::from_vec(real);

        // Reshape 1D into 4D
        let real = real.into_shape(array_to_tuple(shape)).unwrap();

        array_to_disk(real_path, &real).expect("write to disk in parallel failed");

        timer.elapsed().as_millis()
    });
    let imag_handle: std::thread::JoinHandle<_> = spawn(move || {
        // Gather imag values
        let imag: Vec<T> = imag_host.iter().map(|x| x.im).collect();

        // Build nd_array for npy serialization
        let imag: ndarray::Array1<T> = ndarray::ArrayBase::from_vec(imag);

        // Reshape 1D into 4D
        let imag = imag.into_shape(array_to_tuple(shape)).unwrap();

        array_to_disk(imag_path, &imag).expect("write to disk in parallel failed");

        timer.elapsed().as_millis()
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

fn array_to_disk<T>(path: String, array: &ndarray::Array4<T>) -> Result<()>
where
    T: Float + HasAfEnum + WritableElement,
{
    // Write to npy
    write_npy(path, array).expect("write to disk failed");
    Ok(())
}

/// This is a helper function to turn a length 4 array (required by Dim4) into a tuple,
/// which is required by ndarray::Array's .reshape() method
pub fn array_to_tuple(dim4: [u64; 4]) -> (usize, usize, usize, usize) {
    (
        dim4[0] as usize,
        dim4[1] as usize,
        dim4[2] as usize,
        dim4[3] as usize,
    )
}

/// This is a helper function to turn a length 4 array (required by Dim4) into a Dim4,
pub fn array_to_dim4(dim4: [u64; 4]) -> Dim4 {
    Dim4::new(&dim4)
}

/// This function reads toml files
pub fn read_toml<T: DeserializeOwned>(path: &str) -> Result<T, RuntimeError> {
    // Read toml config file
    let toml_contents: &str =
        &std::fs::read_to_string(path).map_err(|_| RuntimeError::TomlReadError {
            path: path.to_string(),
        })?;

    // Return parsed toml from str
    toml::from_str(toml_contents).map_err(|e| RuntimeError::TomlParseError {
        msg: format!("{e:?}"),
    })
}

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
    /// Sampling Parameters
    pub sampling: Option<TomlSamplingParameters>,

    /// Cosmological Parameters
    #[cfg(feature = "expanding")]
    pub cosmology: CosmologyParameters,

    #[cfg(feature = "remote-storage")]
    pub remote_storage_parameters: RemoteStorageParameters,
}

#[derive(Deserialize, Debug, PartialEq)]
pub struct TomlSamplingParameters {
    pub scheme: SamplingScheme,
    #[serde(deserialize_with = "deserialize_seeds")]
    pub seeds: Vec<u64>,
}

pub fn parameters_from_toml<
    T: FromPrimitive
        + Float
        + FloatingPoint
        + Display
        + HasAfEnum<InType = T>
        + HasAfEnum<BaseType = T>
        + Fromf64
        + ConstGenerator<OutType = T>,
>(
    toml: TomlParameters,
) -> Result<Vec<SimulationParameters<T>>, RuntimeError> {
    // Extract required parameters from toml
    let axis_length: T = T::from_f64(toml.axis_length).unwrap();
    let time: T = T::from_f64(toml.time.unwrap_or(0.0)).unwrap();
    #[allow(unused_assignments)]
    let final_sim_time: T = T::from_f64(toml.final_sim_time).unwrap();
    let cfl: T = T::from_f64(toml.cfl).unwrap();
    let num_data_dumps: u32 = toml.num_data_dumps;
    let total_mass: f64 = toml.total_mass;
    let sim_name: String = toml.sim_name;
    let k2_cutoff: f64 = toml.k2_cutoff;
    let alias_threshold: f64 = toml.alias_threshold;
    let dims = num::FromPrimitive::from_usize(toml.dims).unwrap();
    let size = toml.size;

    // Calculate overdetermined parameters
    let particle_mass;
    let hbar_;
    if let Some(ntot) = toml.ntot {
        // User has specified the total mass and ntot.
        // So, the particle mass can be derived.

        particle_mass = toml.total_mass / ntot;
        hbar_ = toml.hbar_.unwrap_or_else(|| {
            println!("hbar_ not specified, using HBAR / particle_mass.");
            HBAR / particle_mass
        });
    } else if let Some(p_mass) = toml.particle_mass {
        // User has specified the total mass and particle mass.
        // So, the ntot can be derived, as can hbar_ if not specified.

        particle_mass = p_mass;
        hbar_ = toml.hbar_.unwrap_or_else(|| {
            println!("hbar_ not specified, using HBAR / particle_mass.");
            HBAR / particle_mass
        });
    } else if let Some(hbar_tilde) = toml.hbar_ {
        // User has specified the total mass and hbar_.
        // So, the ntot and particle_mass can be derived.

        hbar_ = hbar_tilde;
        particle_mass = HBAR / hbar_
        // ntot isn't actually stored but is determined via total_mass / particle_mass;
    } else {
        panic!(
            "You must specify the total mass and either exactly one of ntot (total number \
                  of particles) or particle_mass, or hbar_tilde ( hbar / particle_mass ). Note: you
                  can specify hbar_tilde in addition to one of the first two if you'd like to change
                  the value of planck's constant itself."
        )
    }

    if let Some(TomlSamplingParameters { scheme, seeds }) = toml.sampling {
        // If sampling parametrs are specified, generate a set of simulation parameters for each seed
        Ok(seeds
            .into_iter()
            .map(Some)
            .map(|seed| {
                SimulationParameters::new(
                    T::from_f64(toml.axis_length).unwrap(),
                    toml.time
                        .map(T::from_f64)
                        .unwrap_or(Some(T::zero()))
                        .unwrap(),
                    T::from_f64(toml.final_sim_time).unwrap(),
                    T::from_f64(toml.cfl).unwrap(),
                    num_data_dumps,
                    total_mass,
                    particle_mass,
                    sim_name.clone(),
                    k2_cutoff,
                    alias_threshold,
                    Some(hbar_),
                    dims,
                    size,
                    #[cfg(feature = "expanding")]
                    toml.cosmology,
                    Some(SamplingParameters { seed, scheme }),
                    toml.ics.clone(),
                )
            })
            .collect())
    } else {
        // If no seeds are specified, return only the parameters for the MFT
        Ok(vec![SimulationParameters::<T>::new(
            axis_length,
            time,
            final_sim_time,
            cfl,
            num_data_dumps,
            total_mass,
            particle_mass,
            sim_name,
            k2_cutoff,
            alias_threshold,
            Some(hbar_),
            dims,
            size,
            #[cfg(feature = "expanding")]
            toml.cosmology,
            None,
            toml.ics,
        )])
    }
}

fn deserialize_seeds<'de, D>(deserializer: D) -> Result<Vec<u64>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    // Get the string
    let Ok(parsed_string) = <&str>::deserialize(deserializer) else {
        panic!("fialed to parsed");
    };

    println!("parsed_string = {parsed_string}");

    parse_seeds(parsed_string).map_err(serde::de::Error::custom)
}

#[test]
fn test_regex_range_exclusive() {
    let sample = "0..=55";
    let seeds = parse_seeds(sample);
    assert_eq!(seeds, Ok((0..=55).collect()));
}

#[test]
fn test_regex_to() {
    let sample = "0 to 55";
    let seeds = parse_seeds(sample);
    assert_eq!(seeds, Ok((0..=55).collect()));
}

#[test]
fn test_regex_comma_separated() {
    let sample = "[1, 3]";
    let seeds = parse_seeds(sample);
    assert_eq!(seeds, Ok([1, 3].into_iter().collect()));

    let sample = "1, 3";
    let seeds = parse_seeds(sample);
    assert_eq!(seeds, Ok([1, 3].into_iter().collect()));
}

/// NOTE: this compiles the regex internally,
/// so if we ever use this multiple times it will be slow/inefficient.
fn parse_seeds(s: &str) -> Result<Vec<u64>, &'static str> {
    let r1 = Regex::new(r"\d+..=\d+").unwrap();
    let r2 = Regex::new(r"\d+ to \d+").unwrap();
    let r3 = Regex::new(r"(\d+)[^,]?+").unwrap();
    println!("string is {s}");
    match s {
        // Range Inclusive (a..=b)
        _ if r1.is_match(s) => {
            // Get start and end points
            let [start, end]: [u64; 2] = s
                .split("..=")
                .map(str::parse::<u64>)
                .map(Result::unwrap)
                .collect::<Vec<u64>>()
                .try_into()
                .unwrap();

            // Collect seeds
            Ok((start..=end).collect())
        }

        // Custom syntax (a to b)
        _ if r2.is_match(s) => {
            // Get start and end points
            let [start, end]: [u64; 2] = s
                .split(" to ")
                .map(str::parse::<u64>)
                .map(Result::unwrap)
                .collect::<Vec<u64>>()
                .try_into()
                .unwrap();

            // Collect seeds
            Ok((start..=end).collect())
        }

        // Comma separated digits
        _ if r3.is_match(s) => {
            // Collect and parse collection
            Ok(r3
                .find_iter(s)
                .map(|_match| {
                    println!("{}", _match.as_str().trim_matches(']'));
                    str::parse::<u64>(_match.as_str().trim_matches(']'))
                })
                .map(Result::unwrap)
                .collect::<Vec<u64>>())
        }

        _ => {
            return Err(
                "seeds did not match expected patterns: low..=high, low to high, [s1, s2, s3]",
            )
        }
    }
}

#[test]
fn test_deserialize_toml() {
    let axis_length = 30.0;
    let final_sim_time = 400.0;
    let cfl = 0.5;
    let num_data_dumps = 100;
    let total_mass = 1e10;
    let hbar_ = 0.02;
    let sim_name = "gaussian-overdensity-512-mft";
    let k2_cutoff = 0.95;
    let alias_threshold = 0.02;
    let dims = 3;
    let size = 512;
    let ics = InitialConditions::ColdGaussKSpace {
        mean: vec![15., 15., 15.],
        std: vec![10., 10., 10.],
        phase_seed: None,
    };
    let ics_string = toml::to_string(&ics).unwrap();
    let seeds = 1..=64;
    let scheme = SamplingScheme::Husimi;

    //ics                         = {{ {ics_string} }} \n\n \

    let toml_contents: String = format!(
        "\
    axis_length                 = {axis_length}\n\
    final_sim_time              = {final_sim_time}\n\
    cfl                         = {cfl}\n\
    num_data_dumps              = {num_data_dumps}\n\
    total_mass                  = {total_mass}\n\
    hbar_                       = {hbar_}\n\
    sim_name                    = \"{sim_name}\"\n\
    k2_cutoff                   = {k2_cutoff}\n\
    alias_threshold             = {alias_threshold}\n\
    dims                        = {dims}\n\
    size                        = {size}\n\n\
    \
    [ics]\n\
    {ics_string}\n\n\
    \
    [sampling]\n\
    seeds = \"{seeds:?}\"\n\
    scheme = \"{scheme:?}\"\n\
    "
    );
    println!("{toml_contents}");

    let toml: TomlParameters = toml::from_str(&toml_contents).unwrap();

    assert_eq!(toml.axis_length, axis_length);
    assert_eq!(toml.final_sim_time, final_sim_time);
    assert_eq!(toml.cfl, cfl);
    assert_eq!(toml.num_data_dumps, num_data_dumps);
    assert_eq!(toml.total_mass, total_mass);
    assert_eq!(toml.hbar_, Some(hbar_));
    assert_eq!(toml.sim_name, sim_name);
    assert_eq!(toml.k2_cutoff, k2_cutoff);
    assert_eq!(toml.alias_threshold, alias_threshold);
    assert_eq!(toml.dims, dims);
    assert_eq!(toml.ics, ics);
    assert_eq!(
        toml.sampling,
        Some(TomlSamplingParameters {
            seeds: seeds.collect(),
            scheme
        })
    );
}

#[cfg(feature = "remote-storage")]
#[derive(Deserialize)]
pub struct RemoteStorageParameters {
    /// Keypair to use for remote storage. NOTE: supply a path, and the deserializer will attempt to deserialize
    #[serde(deserialize_with = "deserialize_keypair")]
    #[serde(alias = "keypair_path")]
    pub keypair: Keypair,

    /// Storage account to use
    pub storage_account: String,
}

#[cfg(feature = "remote-storage")]
pub(crate) struct RemoteStorage {
    /// ShadowDrive client
    pub(crate) client: Arc<ShadowDriveClient<Keypair>>,
    /// Tokio runtime
    pub(crate) rt: Runtime,
    /// Storage Account
    pub(crate) storage_account: Pubkey,
    /// Storage account files
    pub(crate) files: Arc<RwLock<Vec<String>>>,
}

#[cfg(feature = "remote-storage")]
impl RemoteStorage {
    pub(crate) fn new(remote_storage_parameters: RemoteStorageParameters) -> RemoteStorage {
        const SOLANA_MAINNET_BETA_ENDPOINT: &'static str = "https://api.mainnet-beta.solana.com";
        // Get the pubkey associated with this keypair
        let pubkey = remote_storage_parameters.keypair.pubkey();

        // Initialize tokio runtime
        let rt = Builder::new_current_thread()
            .enable_io()
            .enable_time()
            .build()
            .unwrap();

        // Initialize client and get storage account pubkey
        let client = Arc::new(ShadowDriveClient::new(
            remote_storage_parameters.keypair,
            SOLANA_MAINNET_BETA_ENDPOINT,
        ));
        let storage_account: Pubkey = get_account_pubkey(
            rt.block_on(client.get_storage_accounts(&pubkey))
                .expect("client error")
                .into_iter()
                .find(|acct| get_account_name(acct) == &remote_storage_parameters.storage_account)
                .expect("storage account doesn't exist"),
        );

        // Get files currently stores on account
        let files = Arc::new(RwLock::new(
            rt.block_on(client.list_objects(&storage_account))
                .expect("failed to obtain file names in remote storage account"),
        ));

        RemoteStorage {
            client,
            rt,
            storage_account,
            files,
        }
    }

    pub(crate) fn upload_grid<T: Float + HasAfEnum + Default + serde::Serialize + std::fmt::Debug>(
        &self,
        grid: &Array<Complex<T>>,
        filename: String,
    ) -> tokio::task::JoinHandle<String>
    where
        Complex<T>: arrayfire::HasAfEnum + serde::Serialize,
    {
        // Serialize the grid
        let bytes: Vec<u8> = bincode::serialize(&grid).unwrap();

        // Upload the grid
        let client = Arc::clone(&self.client);
        let storage_account = self.storage_account.clone();
        let files = Arc::clone(&self.files);
        if files.read().unwrap().contains(&filename) {
            return self.rt.spawn(async move {
                // Overwrite file
                let mut response = client
                    .edit_file(&storage_account, ShadowFile::bytes(filename, bytes))
                    .await
                    .expect("failed to upload file");

                // Return url
                response.finalized_locations.remove(0)
            });
        } else {
            return self.rt.spawn(async move {
                // Upload file
                let mut response = client
                    .store_files(
                        &storage_account,
                        vec![ShadowFile::bytes(filename.clone(), bytes)],
                    )
                    .await
                    .expect("failed to upload file");

                // Add file to files
                files.write().unwrap().push(filename);

                // Return url
                response.finalized_locations.remove(0)
            });
        }
    }
}

#[cfg(feature = "remote-storage")]
fn deserialize_keypair<'de, D>(deserializer: D) -> Result<Keypair, D::Error>
where
    D: serde::Deserializer<'de>,
{
    // The user supplied a path
    let path: PathBuf = match serde::Deserialize::deserialize(deserializer) {
        Ok(s) => s,
        Err(e) => return Err(e),
    };

    read_keypair_file(path).map_err(serde::de::Error::custom)
}

#[cfg(feature = "remote-storage")]
fn get_account_name(acct: &StorageAcct) -> &str {
    match acct {
        &StorageAcct::V1(ref account) => &account.identifier,
        &StorageAcct::V2(ref account) => &account.identifier,
    }
}

#[cfg(feature = "remote-storage")]
fn get_account_pubkey(acct: StorageAcct) -> Pubkey {
    match acct {
        StorageAcct::V1(account) => account.storage_account,
        StorageAcct::V2(account) => account.storage_account,
    }
}
