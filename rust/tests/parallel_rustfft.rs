//! Show how to use an `FFT` object from multiple threads
use std::time::Instant;

#[test]
fn test_parallel_rustfft() {
    let now = Instant::now();

    const FFT_SIZE: usize = 8192;
    extern crate rustfft;

    use std::sync::Arc;
    use std::thread;

    use rustfft::num_complex::Complex32;
    use rustfft::FftPlanner;

    let inverse = false;
    let mut planner = FftPlanner::new();
    let fwd = planner.plan_fft_forward(FFT_SIZE);
    let inv = planner.plan_fft_inverse(FFT_SIZE);

    println!("planners took {}", now.elapsed().as_micros());
    let thread_spawn = Instant::now();
    let threads: Vec<thread::JoinHandle<_>> = (0..2)
        .map(|_| {
            let fwd_copy = Arc::clone(&fwd);

            thread::spawn(move || {
                let mut signal = vec![Complex32::new(0.0, 0.0); FFT_SIZE];
                let mut spectrum = vec![Complex32::new(0.0, 0.0); FFT_SIZE];
                fwd_copy.process(&mut signal);
            })
        })
        .collect();
    println!("spawn threads took {}", thread_spawn.elapsed().as_micros());

    let thread_join = Instant::now();
    for thread in threads {
        thread.join().unwrap();
    }
    println!("join threads took {}", thread_join.elapsed().as_micros());

    println!("total time: {}", now.elapsed().as_micros());
}
