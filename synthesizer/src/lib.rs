// use rayon::prelude::*;
use anyhow::Result;
use dashmap::DashMap;
use glob::glob;
use ndarray::{arr1, Array4, LinalgScalar, ScalarOperand};
use ndarray_npy::{read_npy, write_npy, ReadableElement, WritableElement, WriteNpyError};
use ndrustfft::{Complex, FftNum};
use num::{traits::FloatConst, Float, FromPrimitive, Num, ToPrimitive};
use std::cmp::Eq;
use std::fmt::Display;
use std::hash::Hash as Hashable;
use std::ops::{AddAssign, DivAssign, MulAssign, RemAssign, SubAssign};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread::{spawn, JoinHandle};
use std::time::Instant;

// TODO: Refactor all mpi stuff into balancer
#[cfg(feature = "balancer")]
use mpi::{datatype::PartitionMut, traits::*, Count};
#[cfg(feature = "balancer")]
pub mod balancer;
#[cfg(not(feature = "balancer"))]
pub mod balancer_nompi;
use balancer::Balancer;
use balancer_nompi as balancer;

static CREATE: AtomicBool = AtomicBool::new(false);
macro_rules! debug_check_complex {
    ($z:expr) => {{
        debug_assert!(check_complex($z));
        $z
    }};
}
macro_rules! debug_check {
    ($z:expr) => {{
        debug_assert!(check($z));
        $z
    }};
}

/// This function loads the npy file located at `filepath`,
/// returning a Vec<T> containing that data.
pub fn load_complex<T>(file: String) -> Array4<Complex<T>>
where
    T: Num
        + Float
        + Default
        + Equivalence
        + ToPrimitive
        + ReadableElement
        + Copy
        + Send
        + 'static
        + std::fmt::Debug,
{
    // Read in real/imag fields
    let real_file = format!("{file}_real");
    let imag_file = format!("{file}_imag");

    // Send one I/O operation to another thread, do one with this thread
    let real: JoinHandle<Array4<Complex<T>>> = spawn(move || {
        let real_array: Array4<T> = read_npy(&real_file)
            .map_err(|_| Err::<Array4<T>, String>(format!("couldn't open {real_file}")))
            .unwrap();
        real_array.map(|&x: &T| Complex::<T>::new(x, T::zero()))
    });
    let imag: Array4<T> = read_npy(&imag_file)
        .map_err(|_| Err::<Array4<T>, String>(format!("couldn't open {imag_file}")))
        .unwrap();
    let imag = imag.map(|&x| Complex::<T>::new(T::zero(), x));

    real.join().unwrap() + imag
}

pub fn dump_complex<T>(v: Array4<Complex<T>>, path: String) -> Result<()>
where
    T: Num
        + Float
        + Default
        + Equivalence
        + ToPrimitive
        + WritableElement
        + Copy
        + Send
        + Sync
        + 'static,
{
    let real_path = format!("{path}_real");
    let imag_path = format!("{path}_imag");

    if !CREATE.swap(false, Ordering::Relaxed) {
        drop(std::fs::create_dir_all(
            std::path::PathBuf::from(&real_path).parent().unwrap(),
        ));
    }

    // Thread safe references
    let real_v = Arc::new(v);
    let imag_v = Arc::clone(&real_v);

    // Send one I/O operation to another thread, do one with this thread
    let real: JoinHandle<Result<(), WriteNpyError>> =
        spawn(move || write_npy(real_path, &real_v.map(|x| x.re)));
    let imag: Result<(), WriteNpyError> = write_npy(imag_path, &imag_v.map(|x| x.im));

    real.join().unwrap().unwrap();
    Ok(imag.unwrap())
}

/// Analyze the simulations with base name `sim_base_name`.
pub fn analyze_sims<T, Name>(
    functions: Arc<Functions<T, Name>>,
    sim_base_name: &String,
    dumps: &Vec<u16>,
    balancer: &mut Balancer<()>,
) -> Result<()>
where
    T: Num
        + Float
        + Default
        + Equivalence
        + ToPrimitive
        + ReadableElement
        + WritableElement
        + Copy
        + FromPrimitive
        + FftNum
        + FloatConst
        + LinalgScalar
        + MulAssign
        + SubAssign
        + AddAssign
        + RemAssign
        + DivAssign,
    Complex<T>: ScalarOperand + LinalgScalar + DivAssign<T> + AddAssign<Complex<T>>,
    Name: AsRef<str> + Clone + Eq + Hashable + Display + Send + Sync + 'static + Ord,
{
    {
        // Get local set of dumps
        let local_dumps = balancer.local_set(dumps);
        log::debug!("local_dumps {} = {local_dumps:?}", balancer.rank);

        // Get dims and size for move closures;
        let dims = functions.dims;
        let size = functions.size;

        for &dump in local_dumps.iter() {
            // The number of sims with this dump
            let nsims: Arc<AtomicUsize> = Arc::new(AtomicUsize::new(0));

            // TODO: come back to this to allow resuming partial DI progress
            // let psi_exists =
            //     std::path::Path::new(&format!("{}-combined/psi_{:05}_real", sim_base_name, dump))
            //         .exists();
            // let psi2_exists =
            //     std::path::Path::new(&format!("{}-combined/psi2_{:05}_real", sim_base_name, dump))
            //         .exists();
            // let psik_exists =
            //     std::path::Path::new(&format!("{}-combined/psik_{:05}_real", sim_base_name, dump))
            //         .exists();
            // let psik2_exists = std::path::Path::new(&format!(
            //     "{}-combined/psik2_{:05}_real",
            //     sim_base_name, dump
            // ))
            // .exists();
            // let psi_exists =
            //     std::path::Path::new(&format!("{}-combined/psi_{:05}_imag", sim_base_name, dump))
            //         .exists()
            //         && psi_exists;
            // let psi2_exists =
            //     std::path::Path::new(&format!("{}-combined/psi2_{:05}_imag", sim_base_name, dump))
            //         .exists()
            //         && psi2_exists;
            // let psik_exists =
            //     std::path::Path::new(&format!("{}-combined/psik_{:05}_imag", sim_base_name, dump))
            //         .exists()
            //         && psik_exists;
            // let psik2_exists = std::path::Path::new(&format!(
            //     "{}-combined/psik2_{:05}_imag",
            //     sim_base_name, dump
            // ))
            // .exists()
            //     && psik2_exists;
            // if psi_exists && psi2_exists && psik_exists && psik2_exists {
            //     nsims.fetch_add(1, Ordering::Relaxed);
            //     continue;
            // }

            log::trace!("rank {} is taking on dump {dump}", balancer.rank);
            let sims = glob(format!("./{}-stream*/", sim_base_name).as_str())
                .unwrap()
                .map(Result::unwrap)
                .map(|p| p.display().to_string());

            for sim in sims {
                // Clone the Arc containing the functions to be evaluated
                let local_functions: Arc<Functions<T, Name>> = functions.clone();

                // Clone counter Arc
                let local_counter: Arc<AtomicUsize> = nsims.clone();

                balancer.spawn(move || {
                    // Start timer for this (dump, sim)
                    let now = Instant::now();

                    log::trace!("Starting (dump, sim) = ({dump}, {sim})");

                    // Load psi for this (dump, sim)
                    let ψ = load_complex::<T>(format!("{}/psi_{:05}", &sim, dump));
                    // Calculate fft
                    let mut fft_handler: ndrustfft::FftHandler<T> =
                        ndrustfft::FftHandler::new(size);
                    let mut ψ_buffer;
                    let mut ψk = ψ.clone();
                    for dim in 0..dims {
                        ψ_buffer = ψk.clone();
                        ndrustfft::ndfft(&ψ_buffer, &mut ψk, &mut fft_handler, dim);
                    }
                    assert!(ψk.iter().all(|x| x.re.is_finite() && x.im.is_finite()));

                    // First calculate all fields
                    for key_value in local_functions.array_functions.functions.iter() {
                        // Extract (key, value) pair
                        let (field_name, function) = key_value.pair();

                        // AddAssign function output to entry
                        let add_value = function(&ψ, &ψk);
                        local_functions
                            .array_functions
                            .values
                            .get_mut(field_name)
                            .unwrap()
                            .value_mut()
                            .add_assign(&add_value);

                        // TODO debug
                        debug_assert!(local_functions
                            .array_functions
                            .values
                            .get(field_name)
                            .unwrap()
                            .value()
                            .into_iter()
                            .all(|x| x.re.is_finite() && x.im.is_finite()))
                    }

                    // Then calculate all scalars
                    for key_value in local_functions.scalar_functions.functions.iter() {
                        // Extract (key, value) pair
                        let (field_name, function) = key_value.pair();

                        // AddAssign function output to entry
                        let add_value = function(&ψ, &ψk);
                        local_functions
                            .scalar_functions
                            .values
                            .get_mut(field_name)
                            .unwrap()
                            .value_mut()
                            .add_assign(add_value);
                        debug_check_complex!(local_functions
                            .scalar_functions
                            .values
                            .get(field_name)
                            .unwrap()
                            .value());
                    }

                    // Increment sims counter for this dump
                    local_counter.fetch_add(1, Ordering::Relaxed);

                    log::trace!(
                        "Finished (dump, sim) = ({dump}, {sim}) in {} seconds",
                        now.elapsed().as_secs()
                    );
                });
            }

            // Wait for iteration though all sims (only on this thread)
            log::trace!("waiting");
            balancer.barrier();
            log::info!("rank {} finished dump {dump}", balancer.rank);

            // Average and dump
            let nsims_usize: usize = nsims.load(Ordering::Relaxed);
            assert!(nsims_usize > 0);
            let path_base = format!("{}-combined/PLACEHOLDER_{:05}", sim_base_name, dump);
            {
                // First average and dump all fields (and then zero out)
                for key_value in functions.array_functions.functions.iter() {
                    // Extract (key, value) pair
                    let field_name = key_value.key();

                    // DivAssign function to average
                    let mut entry = functions
                        .array_functions
                        .values
                        .get_mut(field_name)
                        .unwrap();
                    entry
                        .value_mut()
                        .div_assign(Complex::new(T::from_usize(nsims_usize).unwrap(), T::zero()));
                    entry.iter().for_each(|z| {
                        debug_check_complex!(z);
                    });

                    // Write result to disk
                    let write_path = path_base.replace("PLACEHOLDER", field_name.as_ref());
                    dump_complex::<T>(entry.value().clone(), write_path)
                        .expect(format!("Failed to dump {field_name}").as_str());
                }

                // Then average all scalars (and then zero out)
                for key_value in functions.scalar_functions.functions.iter() {
                    // Extract (key, value) pair
                    let field_name = key_value.key();

                    // DivAssign function to average
                    let mut entry = functions
                        .scalar_functions
                        .values
                        .get_mut(field_name)
                        .unwrap();
                    entry
                        .value_mut()
                        .div_assign(T::from_usize(nsims_usize).unwrap());
                    debug_check_complex!(*entry);

                    // Write result to disk
                    let dump_array = arr1(&[*entry.value()])
                        .into_shape((1, 1, 1, 1))
                        .expect("This array should only have one value");
                    dump_complex::<T>(
                        dump_array,
                        path_base.replace("PLACEHOLDER", field_name.as_ref()),
                    )
                    .expect(format!("Failed to dump {field_name}").as_str());
                }
            }

            // Zero out all entries to begin AddAssign on next dump
            functions.zero()
        }
    }

    Ok(())
}

#[cfg(not(feature = "balancer"))]
pub trait Equivalence {
    /* dummy trait */
}

/// Function which analyzes the resulting combined quantities after combining all individual streams
#[allow(unreachable_code)]
pub fn post_combine<T, Name>(
    functions: Arc<PostCombineFunctions<T, Name>>,
    combined_base_name: String,
    dumps: &Vec<u16>,
    balancer: &mut Balancer<()>,
) -> Result<()>
where
    T: Num
        + Float
        + Default
        + ToPrimitive
        + ReadableElement
        + WritableElement
        + Copy
        + FromPrimitive
        + FftNum
        + FloatConst
        + LinalgScalar
        + MulAssign
        + SubAssign
        + AddAssign
        + RemAssign
        + DivAssign
        + Equivalence,
    Complex<T>: ScalarOperand + LinalgScalar + DivAssign<T> + AddAssign<Complex<T>>,
    Name: AsRef<str> + Clone + Eq + Hashable + Display + Send + Sync + 'static + Ord,
{
    {
        // Get local set
        let local_dumps = balancer.local_set(dumps);
        log::debug!("local_dumps {} = {local_dumps:?}", balancer.rank);

        // Put the combined base name in an Arc so that threads can share
        let combined_base_name = Arc::new(combined_base_name);

        for &dump in local_dumps.iter() {
            // Make a clone of the pointer
            let local_combined_base_name = Arc::clone(&combined_base_name);

            // Clone the Arc containing the functions to be evaluated
            let local_functions: Arc<PostCombineFunctions<T, Name>> = functions.clone();

            log::trace!("rank {} is starting {} postcombine", balancer.rank, dump);
            balancer.spawn(move || {
                // Start timer for this (dump, sim)
                let now = Instant::now();

                // Load psis for this (dump, sim)
                let ψ = load_complex::<T>(
                    local_combined_base_name
                        .replace("PLACEHOLDER", "psi")
                        .replace("DUMP", format!("{:05}", dump).as_str()),
                );
                let ψ2 = load_complex::<T>(
                    local_combined_base_name
                        .replace("PLACEHOLDER", "psi2")
                        .replace("DUMP", format!("{:05}", dump).as_str()),
                );
                let ψk = load_complex::<T>(
                    local_combined_base_name
                        .replace("PLACEHOLDER", "psik")
                        .replace("DUMP", format!("{:05}", dump).as_str()),
                );
                let ψk2 = load_complex::<T>(
                    local_combined_base_name
                        .replace("PLACEHOLDER", "psik2")
                        .replace("DUMP", format!("{:05}", dump).as_str()),
                );

                // First calculate all fields
                #[allow(unused)] // we don't use array functions right now
                for key_value in local_functions.post_array_functions.functions.iter() {
                    todo!("need to implement post-combine array function dump");

                    // Extract (key, value) pair
                    let (field_name, function) = key_value.pair();

                    // AddAssign function output to entry
                    let output = function(&ψ, &ψ2, &ψk, &ψk2);

                    // Dump to disk
                    todo!("need to implement post-combine array function dump")
                }

                // Then calculate all scalars
                for key_value in local_functions.post_scalar_functions.functions.iter() {
                    // Extract (key, value) pair
                    let (field_name, function) = key_value.pair();

                    // Calculate scalar output
                    let output = function(dump, &ψ, &ψ2, &ψk, &ψk2);

                    // Push to vector
                    local_functions
                        .post_scalar_functions
                        .values
                        .get_mut(field_name)
                        .unwrap()
                        .value_mut()
                        .push(output);
                }

                log::trace!(
                    "Finished dump {dump} in {} seconds",
                    now.elapsed().as_secs()
                );
            });
        }

        // Wait for iteration though all sims
        // extra debug info: waits for local threads to finish before printing
        log::debug!("rank {} is waiting for local tasks", balancer.rank);
        balancer.wait();

        // this block synchronizes all fields across nodes (mpi enabled)
        #[cfg(feature = "balancer")]
        {
            log::debug!("rank {} is waiting for all others", balancer.rank);
            balancer.barrier();
            if balancer.rank == 0 {
                log::debug!("all done. onto mpi syncronization");
            }

            // Sort fields locally so all ranks agree on order
            let mut fields_sorted: Vec<Name> = functions
                .post_scalar_functions
                .functions
                .iter()
                .map(|x| x.key().clone())
                .collect::<Vec<Name>>();
            fields_sorted.sort();

            // Iterate through scalars and dump
            for (rank, field_name) in fields_sorted.iter().enumerate() {
                // Round robin I/O leader for this scalar
                let leader_rank = (rank % balancer.size) as i32;
                let leader = balancer.world.process_at_rank(leader_rank);

                // Push to vector (cast to f64, then back to T after sending)
                let vec_to_send: Vec<Scalar> = functions
                    .post_scalar_functions
                    .values
                    .get(field_name)
                    .unwrap()
                    .value()
                    .iter()
                    .map(|(dump, value)| Scalar {
                        dump: *dump,
                        re: value.re.to_f64().unwrap(),
                        im: value.im.to_f64().unwrap(),
                    })
                    .collect();

                // Buffer used to gather counts
                let mut gather_counts = vec![0_i32; balancer.size];
                log::debug!("gather_counts({field_name}) is expecting {}", balancer.size);

                // Buffer used to gather Scalar<T>
                let mut gather_vec: Vec<Scalar> = vec![Scalar::default(); dumps.len()];
                log::debug!("gather_vec({field_name}) is expecting {}", dumps.len());

                if balancer.rank == leader_rank as usize {
                    // If leader, first gather counts
                    log::debug!(
                        "rank {} (leader) has {} {}",
                        balancer.rank,
                        vec_to_send.len(),
                        &field_name
                    );
                    leader.gather_into_root(&(vec_to_send.len() as i32), &mut gather_counts[..]);
                    log::debug!("gather_counts = {gather_counts:?}");

                    // Use counts to compute displacements
                    let displs: Vec<Count> = gather_counts
                        .iter()
                        .scan(0, |acc, &x| {
                            let tmp = *acc;
                            *acc += x;
                            Some(tmp)
                        })
                        .collect();
                    log::debug!("displs = {displs:?}");

                    // Then gather payloads
                    {
                        let mut partition =
                            PartitionMut::new(&mut gather_vec[..], gather_counts, &displs[..]);
                        leader.gather_varcount_into_root(&vec_to_send[..], &mut partition);
                        log::debug!("Leader gathered scalar {field_name}: {gather_vec:?}");
                    }

                    // Sort vector
                    gather_vec.sort_by_key(|scalar| scalar.dump);

                    // Map time series into Array4
                    let time_series: Array4<Complex<T>> = arr1(
                        &gather_vec
                            .iter()
                            .map(|scalar| {
                                Complex::<T>::new(
                                    T::from_f64(scalar.re).unwrap(),
                                    T::from_f64(scalar.im).unwrap(),
                                )
                            })
                            .collect::<Vec<Complex<T>>>(),
                    )
                    .into_shape((dumps.len(), 1, 1, 1))
                    .unwrap();
                    log::debug!("Sorted {field_name}: {time_series:?}");

                    // Dump to disk
                    dump_complex::<T, K, S>(
                        time_series,
                        combined_base_name
                            .replace("PLACEHOLDER", field_name.as_ref())
                            .replace("_DUMP", ""),
                    )
                    .expect(format!("Failed to dump {field_name}").as_str());
                } else {
                    // First send count
                    log::debug!(
                        "rank {} is going to send {} {}",
                        balancer.rank,
                        vec_to_send.len(),
                        &field_name
                    );
                    leader.gather_into(&(vec_to_send.len() as i32));

                    // Then send payload
                    leader.gather_varcount_into(&vec_to_send[..]);
                }
            }
        }

        // this block is for one node w/ no mpi
        #[cfg(not(feature = "balancer"))]
        {
            for pair in functions.post_scalar_functions.values.iter() {
                let (field_name, all_dumps) = pair.pair();
                let scalar_series: Vec<Complex<T>> = all_dumps
                    .into_iter()
                    .map(|(_dump, value)| value.clone())
                    .collect();
                let scalar_series =
                    Array4::from_shape_vec((scalar_series.len(), 1, 1, 1), scalar_series).unwrap();
                dump_complex::<T>(
                    scalar_series,
                    combined_base_name
                        .clone()
                        .replace("PLACEHOLDER", field_name.as_ref())
                        .replace("_DUMP", ""),
                )
                .expect(format!("Failed to dump {field_name}").as_str());
            }
        }
    }

    Ok(())
}

#[derive(Default, PartialEq, Debug, Clone)]
#[cfg_attr(feature = "balancer", derive(Equivalence))]
struct Scalar {
    dump: u16,
    re: f64,
    im: f64,
}

#[cfg(not(feature = "balancer"))]
impl<T> Equivalence for T {}

fn get_shape(dims: usize, size: usize) -> (usize, usize, usize, usize) {
    match dims {
        1 => (size, 1, 1, 1),
        2 => (size, size, 1, 1),
        3 => (size, size, size, 1),
        _ => panic!("Invalid dimensions"),
    }
}

type Dump = u16;
pub struct Functions<T, Name>
where
    T: Num
        + Float
        + Default
        + Equivalence
        + ToPrimitive
        + ReadableElement
        + WritableElement
        + Copy
        + FromPrimitive
        + FftNum
        + FloatConst
        + MulAssign
        + SubAssign
        + AddAssign
        + RemAssign,
    Complex<T>: ScalarOperand + MulAssign<Complex<T>> + MulAssign<T>,
    Name: AsRef<str> + Clone + Eq + Hashable + Display + Send + Sync + 'static + Ord,
{
    array_functions: ArrayFunctions<T, Name>,
    scalar_functions: ScalarFunctions<T, Name>,
    dims: usize,
    size: usize,
}

pub struct ArrayFunctions<T, FieldName>
where
    T: Num
        + Float
        + Default
        + Equivalence
        + ToPrimitive
        + ReadableElement
        + WritableElement
        + Copy
        + FromPrimitive
        + FftNum
        + FloatConst,
    Complex<T>: ScalarOperand + MulAssign<Complex<T>> + MulAssign<T>,
    FieldName: AsRef<str> + Clone + Eq + Hashable + Display + Send + Sync + 'static + Ord,
{
    functions: Arc<
        DashMap<
            FieldName,
            Box<
                dyn Fn(&Array4<Complex<T>>, &Array4<Complex<T>>) -> Array4<Complex<T>>
                    + 'static
                    + Send
                    + Sync,
            >,
        >,
    >,
    values: Arc<DashMap<FieldName, Array4<Complex<T>>>>,
}

pub struct ScalarFunctions<T, ScalarName>
where
    T: Num
        + Float
        + Default
        + Equivalence
        + ToPrimitive
        + ReadableElement
        + WritableElement
        + Copy
        + FromPrimitive
        + FftNum
        + FloatConst,
    Complex<T>: ScalarOperand + MulAssign<Complex<T>> + MulAssign<T>,
    ScalarName: AsRef<str> + Clone + Eq + Hashable + Display + Send + Sync + 'static + Ord,
{
    functions: Arc<
        DashMap<
            ScalarName,
            Box<
                dyn Fn(&Array4<Complex<T>>, &Array4<Complex<T>>) -> Complex<T>
                    + 'static
                    + Send
                    + Sync,
            >,
        >,
    >,
    values: Arc<DashMap<ScalarName, Complex<T>>>,
}

pub struct PostCombineFunctions<T, Name>
where
    T: Num
        + Float
        + Default
        + Equivalence
        + ToPrimitive
        + ReadableElement
        + WritableElement
        + Copy
        + FromPrimitive
        + FftNum
        + FloatConst
        + MulAssign
        + SubAssign
        + AddAssign
        + RemAssign,
    Complex<T>: ScalarOperand + MulAssign<Complex<T>> + MulAssign<T>,
    Name: AsRef<str> + Clone + Eq + Hashable + Display + Send + Sync + 'static + Ord,
{
    post_array_functions: PostCombineArrayFunctions<T, Name>,
    post_scalar_functions: PostCombineScalarFunctions<T, Name>,
}

pub struct PostCombineArrayFunctions<T, FieldName>
where
    T: Num
        + Float
        + Default
        + Equivalence
        + ToPrimitive
        + ReadableElement
        + WritableElement
        + Copy
        + FromPrimitive
        + FftNum
        + FloatConst,
    Complex<T>: ScalarOperand + MulAssign<Complex<T>> + MulAssign<T>,
    FieldName: AsRef<str> + Clone + Eq + Hashable + Display + Send + Sync + 'static + Ord,
{
    functions: Arc<
        DashMap<
            FieldName,
            Box<
                dyn Fn(
                        &Array4<Complex<T>>,
                        &Array4<Complex<T>>,
                        &Array4<Complex<T>>,
                        &Array4<Complex<T>>,
                    ) -> Array4<Complex<T>>
                    + 'static
                    + Send
                    + Sync,
            >,
        >,
    >,
    #[allow(dead_code)] // we don't use array functions right now
    values: Arc<DashMap<FieldName, Array4<Complex<T>>>>,
}

pub struct PostCombineScalarFunctions<T, ScalarName>
where
    T: Num
        + Float
        + Default
        + Equivalence
        + ToPrimitive
        + ReadableElement
        + WritableElement
        + Copy
        + FromPrimitive
        + FftNum
        + FloatConst,
    Complex<T>: ScalarOperand + MulAssign<Complex<T>> + MulAssign<T>,
    ScalarName: AsRef<str> + Clone + Eq + Hashable + Display + Send + Sync + 'static + Ord,
{
    functions: Arc<
        DashMap<
            ScalarName,
            Box<
                dyn Fn(
                        Dump,
                        &Array4<Complex<T>>,
                        &Array4<Complex<T>>,
                        &Array4<Complex<T>>,
                        &Array4<Complex<T>>,
                    ) -> (Dump, Complex<T>)
                    + 'static
                    + Send
                    + Sync,
            >,
        >,
    >,
    values: Arc<DashMap<ScalarName, Vec<(u16, Complex<T>)>>>,
}

impl<T, Name> Functions<T, Name>
where
    T: Num
        + Float
        + Default
        + Equivalence
        + ToPrimitive
        + ReadableElement
        + WritableElement
        + Copy
        + FromPrimitive
        + FftNum
        + FloatConst
        + MulAssign
        + SubAssign
        + AddAssign
        + RemAssign,
    Complex<T>: ScalarOperand + MulAssign<Complex<T>> + MulAssign<T>,
    Name: AsRef<str> + Clone + Eq + Hashable + Display + Send + Sync + 'static + Ord,
{
    pub fn new(
        array_functions: Vec<(
            Name,
            Box<
                dyn Fn(&Array4<Complex<T>>, &Array4<Complex<T>>) -> Array4<Complex<T>>
                    + 'static
                    + Send
                    + Sync,
            >,
        )>,
        scalar_functions: Vec<(
            Name,
            Box<
                dyn Fn(&Array4<Complex<T>>, &Array4<Complex<T>>) -> Complex<T>
                    + 'static
                    + Send
                    + Sync,
            >,
        )>,
        dims: usize,
        size: usize,
    ) -> Self {
        // Get shape
        let shape = get_shape(dims, size);

        // Construct array dashmap
        let array_dashmap = Arc::new(DashMap::new());
        let array_values = Arc::new(DashMap::new());
        for (field_name, array_function) in array_functions {
            assert!(
                array_dashmap
                    .insert(field_name.clone(), array_function)
                    .is_none(),
                "Duplicate name for quantity of interest"
            );
            assert!(
                array_values
                    .insert(field_name, Array4::zeros(shape))
                    .is_none(),
                "Duplicate name for quantity of interest"
            );
        }

        // Construct scalar dashmap
        let scalar_dashmap = Arc::new(DashMap::new());
        let scalar_values = Arc::new(DashMap::new());
        for (field_name, scalar_function) in scalar_functions {
            assert!(
                scalar_dashmap
                    .insert(field_name.clone(), scalar_function)
                    .is_none(),
                "Duplicate name for quantity of interest"
            );
            assert!(
                scalar_values
                    .insert(field_name, Complex::new(T::zero(), T::zero()))
                    .is_none(),
                "Duplicate name for quantity of interest"
            );
        }

        Self {
            array_functions: ArrayFunctions {
                functions: array_dashmap,
                values: array_values,
            },
            scalar_functions: ScalarFunctions {
                functions: scalar_dashmap,
                values: scalar_values,
            },
            dims,
            size,
        }
    }

    pub fn zero(&self) {
        // Zero out fields
        for key_value in self.array_functions.functions.iter() {
            // Extract key from (key, value) pair
            let field_name = key_value.key();

            // AddAssign function output to entry
            let mut entry = self.array_functions.values.get_mut(field_name).unwrap();
            *entry.value_mut() = Array4::zeros(get_shape(self.dims, self.size));
        }

        // Zero out scalars
        for key_value in self.scalar_functions.functions.iter() {
            // Extract key from (key, value) pair
            let field_name = key_value.key();

            // AddAssign function output to entry
            let mut entry = self.scalar_functions.values.get_mut(field_name).unwrap();
            *entry.value_mut() *= T::zero();
        }
    }
}

impl<T, Name> PostCombineFunctions<T, Name>
where
    T: Num
        + Float
        + Default
        + Equivalence
        + ToPrimitive
        + ReadableElement
        + WritableElement
        + Copy
        + FromPrimitive
        + FftNum
        + FloatConst
        + MulAssign
        + SubAssign
        + AddAssign
        + RemAssign,
    Name: AsRef<str> + Clone + Eq + Hashable + Display + Send + Sync + 'static + Ord,
    Complex<T>: ScalarOperand + MulAssign<Complex<T>> + MulAssign<T>,
{
    pub fn new(
        post_array_functions: Vec<(
            Name,
            Box<
                dyn Fn(
                        &Array4<Complex<T>>,
                        &Array4<Complex<T>>,
                        &Array4<Complex<T>>,
                        &Array4<Complex<T>>,
                    ) -> Array4<Complex<T>>
                    + 'static
                    + Send
                    + Sync,
            >,
        )>,
        post_scalar_functions: Vec<(
            Name,
            Box<
                dyn Fn(
                        Dump,
                        &Array4<Complex<T>>,
                        &Array4<Complex<T>>,
                        &Array4<Complex<T>>,
                        &Array4<Complex<T>>,
                    ) -> (Dump, Complex<T>)
                    + 'static
                    + Send
                    + Sync,
            >,
        )>,
        dims: usize,
        size: usize,
    ) -> Self {
        // Get shape
        let shape = get_shape(dims, size);

        // Construct array dashmap
        let array_dashmap = Arc::new(DashMap::new());
        let array_values = Arc::new(DashMap::new());
        for (field_name, array_function) in post_array_functions {
            assert!(
                array_dashmap
                    .insert(field_name.clone(), array_function)
                    .is_none(),
                "Duplicate name for quantity of interest"
            );
            assert!(
                array_values
                    .insert(field_name, Array4::zeros(shape))
                    .is_none(),
                "Duplicate name for quantity of interest"
            );
        }

        // Construct scalar dashmap
        let scalar_dashmap = Arc::new(DashMap::new());
        let scalar_values = Arc::new(DashMap::new());
        for (field_name, scalar_function) in post_scalar_functions {
            assert!(
                scalar_dashmap
                    .insert(field_name.clone(), scalar_function)
                    .is_none(),
                "Duplicate name for quantity of interest"
            );
            assert!(
                scalar_values.insert(field_name, vec![]).is_none(),
                "Duplicate name for quantity of interest"
            );
        }

        Self {
            post_array_functions: PostCombineArrayFunctions {
                functions: array_dashmap,
                values: array_values,
            },
            post_scalar_functions: PostCombineScalarFunctions {
                functions: scalar_dashmap,
                values: scalar_values,
            },
        }
    }

    pub fn zero(&self) {
        // // Zero out fields
        // for key_value in self.post_array_functions.functions.iter() {

        //     // Extract key from (key, value) pair
        //     let field_name = key_value.key();

        //     // AddAssign function output to entry
        //     let mut entry = self
        //         .post_array_functions
        //         .values
        //         .get_mut(field_name).unwrap();
        //     *entry.value_mut() = Array4::zeros(get_shape::<K,S>());
        // }

        // Zero out scalars in vec
        for key_value in self.post_scalar_functions.functions.iter() {
            // Extract key from (key, value) pair
            let field_name = key_value.key();

            // Clear vector
            let mut entry = self
                .post_scalar_functions
                .values
                .get_mut(field_name)
                .unwrap();
            entry.value_mut().clear();
        }
    }
}

fn check_complex<T: Float>(z: impl std::borrow::Borrow<Complex<T>>) -> bool {
    z.borrow().re.is_finite() && z.borrow().im.is_finite()
}

fn check<T: Float>(z: impl std::borrow::Borrow<T>) -> bool {
    z.borrow().is_finite()
}
