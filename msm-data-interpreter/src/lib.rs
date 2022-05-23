use rayon::prelude::*;
use num::{Num, Float, FromPrimitive, traits::FloatConst}; 
use ndarray::{Array4, ScalarOperand};
use ndrustfft::{ndfft, Complex, FftNum};
use ndarray_npy::{NpzReader, NpzWriter, ReadableElement, WritableElement};
use std::fs::File;
use std::io::Read;
use anyhow::Result;
use glob::glob;
use std::time::Instant;


/// This function loads the npy file located at `filepath`,
/// returning a Vec<T> containing that data.
pub fn load_complex<T>(
    file: String
) -> Array4<Complex<T>>
where
    T: Num + Float + ReadableElement + Copy,
{
    // Fill buffer vector with data
    let mut npz = NpzReader::new(
        File::open(file).expect("failed to open file")
    ).expect("failed to read npz");

    let real: Array4<T> = npz.by_name("real").unwrap();
    let imag: Array4<T> = npz.by_name("imag").unwrap();

    real.map(|&x| Complex::<T>::new(x, T::zero())) 
    + imag.map(|&x| Complex::<T>::new(T::zero(), x))
}

pub fn dump_complex<T, const K: usize, const S: usize>(
    v: Array4<Complex<T>>,
    path: String,
) -> Result<()>
where
    T: Num + Float + WritableElement + Copy,
{
    println!("writing to {}", &path);
    let mut npz = NpzWriter::new(File::create(path).unwrap());
    npz.add_array("real", &v.map(|x| x.re))?;
    npz.add_array("imag", &v.map(|x| x.im))?;
    npz.finish()?;
    Ok(())
}

/// Analyze the simulations with base name `sim_base_name`.
pub fn analyze_sims<T, const K: usize, const S: usize>(
    sim_base_name: String,
    dump: usize,
) -> Result<()>
where
    T: Num + Float + ReadableElement + WritableElement + Copy + FromPrimitive + FftNum + FloatConst,
    Complex<T>: ScalarOperand,
{
    // Array size
    let size = S.pow(K as u32);

    // TODO: get all dumps given simdirs
    //let dumps = 0..=ndumps;

    // let _dumps_done = dumps.par_bridge().clone().map(|dump| {
    // rayon::scope(|s| {
    //     s.spawn(|_|
        //handler
    //for dump in 1..=ndumps {
    {
        let mut fft_handler: ndrustfft::FftHandler<T> = ndrustfft::FftHandler::new(S);

        // Define buffers
        let mut ψ: Array4<Complex<T>> = ndarray::ArrayBase::zeros(get_shape::<K, S>());
        let mut ψ2: Array4<Complex<T>> = ndarray::ArrayBase::zeros(get_shape::<K, S>());
        let mut ψk: Array4<Complex<T>> = ndarray::ArrayBase::zeros(get_shape::<K, S>());
        let mut ψk2: Array4<Complex<T>> = ndarray::ArrayBase::zeros(get_shape::<K, S>());

        let mut nsims = 0;
        for sim in glob(format!("{}-stream*", sim_base_name).as_str()).unwrap() {
            match sim {
                Ok(sim) => {

                    println!("working on sim {}", sim.display());
                    let now = Instant::now();

                    // Load data dump
                    let filename = format!("{}/psi_{:05}", sim.display(), dump);
                    println!("filename is {}", &filename);
                    let mut data = load_complex::<T>(filename);

                    // Add to wavefunction
                    ψ = ψ + data.clone();

                    // Add to psi squared
                    ψ2 = ψ2 + data.clone()*data.clone().map(|x| x.conj());
                    
                    // Do fft
                    let mut datak = data.clone();
                    for dim in 0..K {
                        if dim > 0 { datak = data.clone(); }
                        ndfft(&data, &mut datak, &mut fft_handler, dim);
                        data = datak.clone();
                    }
                    // Add to wavefunction
                    ψk = ψk + datak.clone();

                    // Add to psi squared
                    ψk2 = ψk2 + datak.clone()*datak.clone().map(|x| x.conj());
                    nsims += 1;
                    println!("Finished sim {} in {} seconds", sim.display(), now.elapsed().as_secs());
                },
                Err(_e) => {}
            }
        }

        // Divide
        ψ   = ψ   / Complex::<T>::new(T::from_usize(nsims).unwrap(), T::zero());
        ψ2  = ψ2  / Complex::<T>::new(T::from_usize(nsims).unwrap(), T::zero());
        ψk  = ψk  / Complex::<T>::new(T::from_usize(nsims).unwrap(), T::zero());
        ψk2 = ψk2 / Complex::<T>::new(T::from_usize(nsims).unwrap(), T::zero());

        // Save
        let dir_path = format!("{}-combined", sim_base_name);
        let path1 = format!("{}-combined/psi_{:05}", sim_base_name, dump);
        let path2 = format!("{}-combined/psi2_{:05}", sim_base_name, dump);
        let path3 = format!("{}-combined/psik_{:05}", sim_base_name, dump);
        let path4 = format!("{}-combined/psik2_{:05}", sim_base_name, dump);
        std::fs::create_dir_all(&dir_path).expect("i/o error");
        dump_complex::<T, K, S>(ψ, path1).expect("i/o error");
        dump_complex::<T, K, S>(ψ2, path2).expect("i/o error");
        dump_complex::<T, K, S>(ψk, path3).expect("i/o error");
        dump_complex::<T, K, S>(ψk2, path4).expect("i/o error");
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