// use rayon::prelude::*;
use num::{Num, Float, FromPrimitive, traits::FloatConst}; 
use ndarray::{Array4, LinalgScalar, ArrayBase, arr1, OwnedRepr, Dim, ScalarOperand};
use ndrustfft::{ndfft, FftNum, Complex, FftHandler};
use ndarray_npy::{read_npy, write_npy, ReadableElement, WritableElement, WriteNpyError};
use std::fs::File;
use anyhow::Result;
use glob::glob;
use mpi::*;
use std::time::Instant;
use std::thread::{spawn, JoinHandle};
use std::sync::{Arc, RwLock};
use std::sync::atomic::{AtomicUsize, Ordering};
use dashmap::DashMap;
use std::ops::{DivAssign, AddAssign, MulAssign, DerefMut, SubAssign, RemAssign};

mod balancer;
use balancer::Balancer;

/// This function loads the npy file located at `filepath`,
/// returning a Vec<T> containing that data.
pub fn load_complex<T>(
    file: String
) -> Array4<Complex<T>>
where
    T: Num + Float + ReadableElement + Copy + Send + 'static,
{
    // Read in real/imag fields
    let real_file = file.clone();
    let real: JoinHandle<Array4<T>> = spawn(move || read_npy(real_file).unwrap());
    let imag: JoinHandle<Array4<T>> = spawn(move || read_npy(file.replace("_real", "_imag")).unwrap());

    real.join().unwrap().map(|&x| Complex::<T>::new(x, T::zero())) 
    + imag.join().unwrap().map(|&x| Complex::<T>::new(T::zero(), x))
}

pub fn dump_complex<T, const K: usize, const S: usize>(
    v: Array4<Complex<T>>,
    path: String,
) -> Result<()>
where
    T: Num + Float + WritableElement + Copy + Send + Sync + 'static,
{
    println!("writing to {}", &path);
    let real_path = format!("{path}_real");
    let imag_path = format!("{path}_imag");

    // Thread safe references
    let real_v = Arc::new(v);
    let imag_v = real_v.clone();

    let real: JoinHandle<Result<(), WriteNpyError> > = spawn(move || write_npy(real_path, &real_v.map(|x| x.re)));
    let imag: JoinHandle<Result<(), WriteNpyError> > = spawn(move || write_npy(imag_path, &imag_v.map(|x| x.im)));

    real.join().unwrap().unwrap();
    imag.join().unwrap().unwrap();

    Ok(())
}

/// Analyze the simulations with base name `sim_base_name`.
pub fn analyze_sims<T, const K: usize, const S: usize>(
    functions: Arc<Functions<T, K, S>>,
    sim_base_name: String,
    dumps: &'static [u16],
) -> Result<()>
where
    T: Num + Float + ReadableElement + WritableElement + Copy + FromPrimitive + FftNum + FloatConst + LinalgScalar + MulAssign + SubAssign + AddAssign + RemAssign + DivAssign,
    Complex<T>: ScalarOperand + LinalgScalar + DivAssign<T> + AddAssign<Complex<T>>,
    Arc<Functions<T, K, S>>: Send,
    ArrayBase<OwnedRepr<Complex<T>>, Dim<[usize; 4]>>: AddAssign + AddAssign<Complex<T>> + DivAssign<Complex<T>> + ScalarOperand,
{
    {

        // Initialize mpi
        let universe = mpi::initialize().expect("Failed to initialize mpi");
        let world = universe.world();
        let mut balancer = Balancer::<()>::new_from_world(world);

        // Get local set
        let local_dumps = balancer.local_set(dumps);

        for dump in local_dumps {

            let nsims: Arc<AtomicUsize> = Arc::new(AtomicUsize::new(0));

            for sim in glob(format!("{}-stream*", sim_base_name).as_str()).unwrap() {

                // Clone the Arc containing the functions to be evaluated
                let local_functions: Arc<Functions<T, K, S>> = functions.clone();

                // Clone counter Arc
                let local_counter: Arc<AtomicUsize> = nsims.clone();

                match sim {
                    Ok(sim) => {
                        println!("working on sim {}", sim.display());

                        balancer.add(
                            spawn(move || {

                                // Start timer for this (dump, sim)
                                let now = Instant::now();

                                // Load psi for this (dump, sim)
                                let ψ = load_complex::<T>(format!("{}/psi_{:05}", sim.display(), dump));

                                // First calculate all fields
                                for key_value in local_functions.array_functions.functions.iter() {
                                    
                                    // Extract (key, value) pair
                                    let (field_name, function) = key_value.pair();

                                    // AddAssign function output to entry
                                    let mut entry = local_functions
                                        .array_functions
                                        .values
                                        .get_mut(field_name).unwrap();
                                    entry.value_mut().add_assign(&function(&ψ));

                                }

                                // Then calculate all scalars
                                for key_value in local_functions.scalar_functions.functions.iter() {
                                    
                                    // Extract (key, value) pair
                                    let (field_name, function) = key_value.pair();

                                    // AddAssign function output to entry
                                    let mut entry = local_functions
                                        .scalar_functions
                                        .values
                                        .get_mut(field_name).unwrap();
                                    entry.value_mut().add_assign(function(&ψ));

                                }

                                // Increment sims counter for this dump
                                local_counter.fetch_add(1, Ordering::SeqCst);

                                println!("Finished sim {} in {} seconds", sim.display(), now.elapsed().as_secs());
                            }));
                    },
                    Err(e) => { panic!("invalid sim file or path: {e:?} ") }
                }
            }

            // Wait for iteration though all sims
            balancer.wait();

            // Average and dump
            let nsims_usize: usize = nsims.load(Ordering::SeqCst);
            let path_base = format!("{}-combined/PLACEHOLDER_{:05}", sim_base_name, dump);
            {
                // First average and dump all fields
                for key_value in functions.array_functions.functions.iter() {
        
                    // Extract (key, value) pair
                    let field_name = key_value.key();

                    // AddAssign function output to entry
                    let mut entry = functions
                        .array_functions
                        .values
                        .get_mut(field_name).unwrap();
                    entry.value_mut().div_assign(Complex::new(T::from_usize(nsims_usize).unwrap(), T::zero()));
                    dump_complex::<T, K, S>(entry.value().clone(), path_base.replace("PLACEHOLDER", field_name)).expect(format!("Failed to dump {field_name}").as_str());

                }

                // Then average all scalars
                for key_value in functions.scalar_functions.functions.iter() {
                    
                    // Extract (key, value) pair
                    let field_name = key_value.key();

                    // AddAssign function output to entry
                    let mut entry = functions
                        .scalar_functions
                        .values
                        .get_mut(field_name).unwrap();
                    entry.value_mut().div_assign(T::from_usize(nsims_usize).unwrap());
                    let dump_array = arr1(&[*entry.value()]).into_shape((1,1,1,1)).expect("This array should only have one value");
                    dump_complex::<T, K, S>(dump_array, path_base.replace("PLACEHOLDER", field_name)).expect(format!("Failed to dump {field_name}").as_str());
                }
            }

        }
        // // Divide
        // (*ψ.write().unwrap()).div_assign(Complex::<T>::new(T::from_usize(nsims.load(Ordering::Acquire)).unwrap(), T::zero()));
        // (*ψ2.write().unwrap()).div_assign(Complex::<T>::new(T::from_usize(nsims.load(Ordering::Acquire)).unwrap(), T::zero()));
        // (*ψk.write().unwrap()).div_assign(Complex::<T>::new(T::from_usize(nsims.load(Ordering::Acquire)).unwrap(), T::zero()));
        // (*ψk2.write().unwrap()).div_assign(Complex::<T>::new(T::from_usize(nsims.load(Ordering::Acquire)).unwrap(), T::zero()));

        // // Save
        // let dir_path = format!("{}-combined", sim_base_name);
        // let path1 = format!("{}-combined/psi_{:05}", sim_base_name, dump);
        // let path2 = format!("{}-combined/psi2_{:05}", sim_base_name, dump);
        // let path3 = format!("{}-combined/psik_{:05}", sim_base_name, dump);
        // let path4 = format!("{}-combined/psik2_{:05}", sim_base_name, dump);
        // std::fs::create_dir_all(&dir_path).expect("i/o error");
        // dump_complex::<T, K, S>(ψ, path1).expect("i/o error");
        // dump_complex::<T, K, S>(ψ2, path2).expect("i/o error");
        // dump_complex::<T, K, S>(ψk, path3).expect("i/o error");
        // dump_complex::<T, K, S>(ψk2, path4).expect("i/o error");
    }

    Ok(())
}

fn get_shape<const K: usize, const S: usize>() -> (usize, usize, usize, usize) {

    match K {
        1 => (S, 1, 1, 1),
        2 => (S, S, 1, 1),
        3 => (S, S, S, 1),
        _ => panic!("Invalid dimensions")
    }
}

// use rayon::prelude::*;
// let sums = [(0, 1), (5, 6), (16, 2), (8, 9)]
//            .par_iter()        // iterating over &(i32, i32)
//            .cloned()          // iterating over (i32, i32)
//            .reduce(|| (0, 0), // the "identity" is 0 in both columns
//                    |a, b| (a.0 + b.0, a.1 + b.1));
// assert_eq!(sums, (0 + 5 + 16 + 8, 1 + 6 + 2 + 9));

type FieldName = String;
type ScalarName = String;
pub struct Functions<T, const K: usize, const S: usize>
where
    T: Num + Float + ReadableElement + WritableElement + Copy + FromPrimitive + FftNum + FloatConst + MulAssign + SubAssign + AddAssign + RemAssign,
    Complex<T>: ScalarOperand + MulAssign<Complex<T>> + MulAssign<T>,
    ArrayBase<OwnedRepr<Complex<T>>, Dim<[usize; 4]>>: AddAssign + AddAssign<Complex<T>> + DivAssign<Complex<T>> + ScalarOperand,
{
    array_functions: ArrayFunctions<T>,
    scalar_functions: ScalarFunctions<T>,
}

pub struct ArrayFunctions<T>
where
    T: Num + Float + ReadableElement + WritableElement + Copy + FromPrimitive + FftNum + FloatConst,
    Complex<T>: ScalarOperand + MulAssign<Complex<T>> + MulAssign<T>,
    ArrayBase<OwnedRepr<Complex<T>>, Dim<[usize; 4]>>: ScalarOperand + AddAssign + AddAssign<Complex<T>> + DivAssign<Complex<T>>,
{
    functions: Arc<DashMap<FieldName, Box<dyn Fn(&Array4<Complex<T>>) -> Array4<Complex<T>> + 'static + Send + Sync>>> ,
    values: Arc<DashMap<FieldName, Array4<Complex<T>>>>,
}

pub struct ScalarFunctions<T>
where
    T: Num + Float + ReadableElement + WritableElement + Copy + FromPrimitive + FftNum + FloatConst,
    Complex<T>: ScalarOperand + MulAssign<Complex<T>> + MulAssign<T>,
    ArrayBase<OwnedRepr<Complex<T>>, Dim<[usize; 4]>>: ScalarOperand + AddAssign + AddAssign<Complex<T>> + DivAssign<Complex<T>>,
{
    functions: Arc<DashMap<ScalarName, Box<dyn Fn(&Array4<Complex<T>>) -> Complex<T> + 'static + Send + Sync>>>,
    values: Arc<DashMap<ScalarName, Complex<T>>>,
}


impl<T, const K: usize, const S: usize> Functions<T, K, S> 
where
    T: Num + Float + ReadableElement + WritableElement + Copy + FromPrimitive + FftNum + FloatConst + MulAssign + SubAssign + AddAssign + RemAssign,
    Complex<T>: ScalarOperand + MulAssign<Complex<T>> + MulAssign<T>,
    ArrayBase<OwnedRepr<Complex<T>>, Dim<[usize; 4]>>: AddAssign + AddAssign<Complex<T>> + DivAssign<Complex<T>> + ScalarOperand,

{

    pub fn new(
        array_functions: Vec<(FieldName, Box<dyn Fn(&Array4<Complex<T>>) -> Array4<Complex<T>> +'static + Send + Sync>)>,
        scalar_functions: Vec<(ScalarName, Box<dyn Fn(&Array4<Complex<T>>) -> Complex<T> + 'static + Send + Sync>)>
    ) -> Self {

        // Get shape
        let shape = get_shape::<K, S>();
        
        // Construct array dashmap
        let array_dashmap = Arc::new(DashMap::new());
        let array_values = Arc::new(DashMap::new());
        for (field_name, array_function) in array_functions {
            assert!(array_dashmap.insert(field_name.clone(), array_function).is_none(), "Duplicate name for quantity of interest");
            
            assert!(array_values.insert(field_name,Array4::zeros(shape)).is_none(), "Duplicate name for quantity of interest");
        }
        

        // Construct scalar dashmap
        let scalar_dashmap = Arc::new(DashMap::new());
        let scalar_values = Arc::new(DashMap::new());
        for (field_name, scalar_function) in scalar_functions {
            assert!(scalar_dashmap.insert(field_name.clone(), scalar_function).is_none(), "Duplicate name for quantity of interest");
            assert!(scalar_values.insert(field_name,Complex::new(T::zero(), T::zero())).is_none(), "Duplicate name for quantity of interest");
        }

        Self {
            array_functions: ArrayFunctions{
                functions: array_dashmap,
                values: array_values,
            },
            scalar_functions: ScalarFunctions{
                functions: scalar_dashmap,
                values: scalar_values,
            },
        }
    }

    pub fn zero(&mut self) {

        // Zero out fields
        for key_value in self.array_functions.functions.iter() {
                                    
            // Extract key from (key, value) pair
            let field_name = key_value.key();

            // AddAssign function output to entry
            let mut entry = self
                .array_functions
                .values
                .get_mut(field_name).unwrap();
            *entry.value_mut() = Array4::zeros(get_shape::<K,S>());

        }

        // Zero out scalars
        for key_value in self.scalar_functions.functions.iter() {
                                    
            // Extract key from (key, value) pair
            let field_name = key_value.key();

            // AddAssign function output to entry
            let mut entry = self
                .scalar_functions
                .values
                .get_mut(field_name).unwrap();
            *entry.value_mut() *= T::zero();

        }
    }
}
