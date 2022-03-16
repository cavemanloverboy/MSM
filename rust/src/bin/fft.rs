use arrayfire::*;
use num::Complex;
use std::time::Instant;

fn main() {

    set_device(0);
    info();
    println!("{:?}", device_mem_info());

    const FFT_SIZE: u64 = 512;
    type T = f32;

    // let array: Array<Complex<T>> = constant!(Complex::<T>::new(0.0, 2.0); FFT_SIZE, FFT_SIZE, FFT_SIZE);
    let now = Instant::now();
    const SIZE: usize = FFT_SIZE.pow(3) as usize;
    let values = vec![Complex::<T>::new(0.0, 2.0); SIZE];
    // for (i, v) in (0..).zip(&mut values) {
    //     *v = Complex::<T>::new(i as T,-1.0*i as T);
    // }
    let dims = Dim4::new(&[FFT_SIZE, FFT_SIZE, FFT_SIZE, 1]);
    let mut array = Array::new(&values, dims);
    println!("3D init array took {} micros", now.elapsed().as_micros());
    //af_print!("Output", array);
    let now = Instant::now();
    let nffts = 1000;
    for _ in 0..nffts {
        fft3_inplace(&mut array, 1.0/(FFT_SIZE as f64).powf(3.0/2.0));
        //af_print!("Output", array);
        ifft3_inplace(&mut array, 1.0/(FFT_SIZE as f64).powf(3.0/2.0));
        //af_print!("Output", array);
    }
    println!("Performed {0} ({1} x {1} x {1}) 3D forward/inverse fft in {2} micros = {3} millis", nffts, FFT_SIZE, now.elapsed().as_micros(), now.elapsed().as_millis());
}
