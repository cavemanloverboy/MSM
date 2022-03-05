use ndarray::arr1;
use rustfft::{num_complex::Complex, FftPlanner};

#[test]
fn test_fft() {
    // Perform a forward FFT of size FFT_SIZE
    const FFT_SIZE: usize = 16;
    type T = f32;

    // Plan forward and inverse ffts
    let mut planner = FftPlanner::<T>::new();
    let fft = planner.plan_fft_forward(FFT_SIZE);
    let ifft = planner.plan_fft_inverse(FFT_SIZE);

    // Define data, save copy for comparison
    let mut buffer = arr1(&[Complex::<T> { re: 1.0, im: 0.0 }; FFT_SIZE]);
    let orig = buffer.clone();
    println!("{:?}", buffer);

    // Do forward FFT, and divide by FFT_size
    fft.process(buffer.as_slice_mut().unwrap());
    buffer = buffer / (FFT_SIZE as T); //.iter().map(| Complex{re:x, im:y}| Complex{re: *x/FFT_SIZE as T, im: *y/FFT_SIZE as T}).collect();
    println!("{:?}", buffer);

    // Do inverse FFT and assert it is close to original
    ifft.process(buffer.as_slice_mut().unwrap());
    println!("{:?}", buffer);
    assert!(
        orig.iter()
            .zip(buffer)
            .fold(0.0, |acc, (x, y)| acc + (x.re - y.re).abs())
            < 1.0e-6
    );
}
