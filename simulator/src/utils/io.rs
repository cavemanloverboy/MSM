use anyhow::Result;
use arrayfire::{Array, ConstGenerator, Dim4, FloatingPoint, Fromf64, HasAfEnum};
use ndarray_npy::{write_npy, WritableElement};
use num::{Complex, Float, FromPrimitive};
use std::fmt::Display;
use std::sync::Arc;
use std::thread::{spawn, JoinHandle};
use std::time::Instant;

use msm_common::{determine_pmass_hbar_, TomlParameters};

use super::error::RuntimeError;
use crate::simulation_object::SimulationParameters;
use msm_common::{SamplingParameters, TomlSamplingParameters};

#[cfg(feature = "remote-storage")]
use {
    msm_common::RemoteStorageParameters,
    shadow_drive_sdk::{
        models::{storage_acct::StorageAcct, ShadowFile},
        ShadowDriveClient,
    },
    solana_sdk::{
        pubkey::Pubkey,
        signature::read_keypair_file,
        signer::{keypair::Keypair, Signer},
    },
    std::sync::RwLock,
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
    // Calculate overdetermined parameters
    let (particle_mass, hbar_) = determine_pmass_hbar_(&toml);

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
    let output_potential = toml.output_potential;

    if let Some(TomlSamplingParameters { scheme, seeds }) = toml.sampling {
        // If sampling parametrs are specified, generate a set of simulation parameters for each seed
        Ok(seeds
            .into_iter()
            .map(|seed| -> Result<SimulationParameters<T>, RuntimeError> {
                Ok(SimulationParameters::new(
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
                    format!("{sim_name}-stream{:05}", seed),
                    k2_cutoff,
                    alias_threshold,
                    Some(hbar_),
                    dims,
                    size,
                    output_potential,
                    #[cfg(feature = "expanding")]
                    toml.cosmology,
                    Some(SamplingParameters { seed, scheme }),
                    toml.ics.clone(),
                    #[cfg(feature = "remote-storage")]
                    RemoteStorage::new(toml.remote_storage_parameters.clone(), Some(seed))?,
                ))
            })
            .collect::<Result<Vec<SimulationParameters<T>>, RuntimeError>>()?)
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
            output_potential,
            #[cfg(feature = "expanding")]
            toml.cosmology,
            None,
            toml.ics,
            #[cfg(feature = "remote-storage")]
            RemoteStorage::new(toml.remote_storage_parameters, None)?,
        )])
    }
}

#[test]
fn test_deserialize_toml() {
    use msm_common::{InitialConditions, SamplingScheme};
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

    #[allow(unused_mut)]
    let mut toml_contents: String = format!(
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
    #[cfg(feature = "remote-storage")]
    toml_contents.push_str(
        " \
    [remote_storage_parameters]
    keypair = \"abc.json\"
    storage_account = \"my account\"
    ",
    );

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

// #[cfg(feature = "remote-storage")]
// impl Clone for RemoteStorageParameters {
//     fn clone(&self) -> Self {
//         RemoteStorageParameters {
//             keypair: Keypair::from_bytes(self.keypair.to_bytes().as_ref()).unwrap(),
//             storage_account: self.storage_account.clone(),
//         }
//     }
// }

#[cfg(feature = "remote-storage")]
pub struct RemoteStorage {
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
    pub(crate) fn new(
        remote_storage_parameters: RemoteStorageParameters,
        stream: Option<u64>,
    ) -> Result<RemoteStorage, RuntimeError> {
        const SOLANA_MAINNET_BETA_ENDPOINT: &'static str = "https://api.mainnet-beta.solana.com";

        // Attempt to retrieve Keypair
        let keypair: Keypair = read_keypair_file(&remote_storage_parameters.keypair)
            .map_err(|e| RuntimeError::KeypairError { e })?;

        // Get the pubkey associated with this keypair
        let pubkey = keypair.pubkey();

        // Initialize tokio runtime
        let rt = Builder::new_current_thread()
            .enable_io()
            .enable_time()
            .build()
            .unwrap();

        // Initialize client and get storage account pubkey
        let client = Arc::new(ShadowDriveClient::new(
            keypair,
            SOLANA_MAINNET_BETA_ENDPOINT,
        ));
        let storage_account: Pubkey = get_account_pubkey({
            let mut storage_accounts: Vec<StorageAcct> = rt
                .block_on(client.get_storage_accounts(&pubkey))
                .expect("client error");
            storage_accounts.retain(|acct| {
                get_account_name(acct).contains(&remote_storage_parameters.storage_account)
            });
            std::panic::catch_unwind(move || {
                if let Some(stream) = stream {
                    // Rotate if doing streams
                    storage_accounts.swap_remove(stream as usize % storage_accounts.len())
                } else {
                    // Otherwise just get first
                    storage_accounts.swap_remove(0)
                }
            })
            .expect("storage account not found")
        });

        // Get files currently stores on account
        let files = Arc::new(RwLock::new(
            rt.block_on(client.list_objects(&storage_account))
                .expect("failed to obtain file names in remote storage account"),
        ));

        Ok(RemoteStorage {
            client,
            rt,
            storage_account,
            files,
        })
    }

    pub(crate) fn upload_grid<T: Float + HasAfEnum + Default + serde::Serialize + std::fmt::Debug>(
        &self,
        grid: &Array<Complex<T>>,
        filename: String,
    ) -> tokio::task::JoinHandle<String>
    where
        Complex<T>: arrayfire::HasAfEnum + serde::Serialize + Send + Sync + 'static,
    {
        // Serialize the grid
        let mut host: Vec<Complex<T>> = vec![Complex::<T>::default(); grid.elements()];
        grid.host(&mut host);

        // Upload the grid
        let client = Arc::clone(&self.client);
        let storage_account = self.storage_account.clone();
        let files = Arc::clone(&self.files);
        if files.read().unwrap().contains(&filename) {
            return self.rt.spawn(async move {
                let bytes: Vec<u8> = bincode::serialize(&host).unwrap();
                // Overwrite file
                let url = format!(
                    "{}/{}/{}",
                    shadow_drive_sdk::constants::SHDW_DRIVE_OBJECT_PREFIX,
                    storage_account,
                    filename,
                );
                let mut response = client
                    .store_files(&storage_account, vec![ShadowFile::bytes(url, bytes)])
                    .await
                    .expect("failed to upload file");

                // Return url
                response.finalized_locations.remove(0)
            });
        } else {
            return self.rt.spawn(async move {
                let bytes: Vec<u8> = bincode::serialize(&host).unwrap();

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
