use msm::utils::FftObject;
use clap::Parser;
use std::time::Instant;
use rustfft::num_complex::Complex;
use ndarray::{Array, Ix3};
use approx::assert_abs_diff_eq;

fn main() {

    // Command line arguments
    let args = Args::parse();

    // Configur rayon ThreadPool
    rayon::ThreadPoolBuilder::new().num_threads({println!("using {} threads", args.threads); args.threads.into()}).build_global().unwrap();

    // Set FFT parameters
    const FFT_SIZE: usize = 256;
    const DIM: usize = 3;
    type T = f64;

    // Create FFT Object
    let fft = FftObject::<T, DIM, FFT_SIZE>::new();

    // Define data to operate on
    // Much easier to test on non-uniform data w/ no symmetries
    let mut data: Array<Complex<T>, Ix3> = Array::from_elem((FFT_SIZE, FFT_SIZE, FFT_SIZE), Complex::<T> { re: 1.0, im: 0.0 });
    data[[0, 0, 0]] = Complex::<T> { re: 1.0, im: 0.0 };
    data[[0, 0, 1]] = Complex::<T> { re: 2.0, im: 0.0 };
    data[[0, 1, 0]] = Complex::<T> { re: 3.0, im: 0.0 };
    data[[0, 1, 1]] = Complex::<T> { re: 4.0, im: 0.0 };
    data[[1, 0, 0]] = Complex::<T> { re: 5.0, im: 0.0 };
    data[[1, 0, 1]] = Complex::<T> { re: 6.0, im: 0.0 };
    data[[1, 1, 0]] = Complex::<T> { re: 7.0, im: 0.0 };
    data[[1, 1, 1]] = Complex::<T> { re: 8.0, im: 0.0 };
    let orig = data.clone();

    // Carry out fwd + inv FFT
    let now = Instant::now();
    fft.forward(&mut data).expect("failed forward");
    println!("{} ms to do forward", now.elapsed().as_millis());

    let now = Instant::now();
    fft.inverse(&mut data).expect("failed inverse");
    println!("{} ms to do inverse", now.elapsed().as_millis());

    // Check that sum of norm of elementwise difference is tiny or zero
    assert_abs_diff_eq!(
        data.map(|x| x.norm()).sum(),
        orig.map(|x| x.norm()).sum(),
        epsilon = 1e-9
    );

}



#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct Args {

    /// Number of rayon threads
    #[clap(short, long, default_value_t = 8)]
    threads: u8,
}