use arrayfire::*;
use num::Complex;
use std::time::Instant;

fn main() {

    set_device(0);
    info();
    println!("{:?}", device_mem_info());

    const FFT_SIZE: u64 = 512;
    let dims = Dim4::new(&[FFT_SIZE, 1, 1, 1]);
    type T = f32;

    // let values = vec![Complex::<T>::new(0.0, 2.0); FFT_SIZE as usize];

    // let signal = Array::new(&values, dims);

    // //af_print!("signal", signal);

    // // Used length of input signal as norm_factor
    // let now = Instant::now();
    // let output = fft(&signal, 0.1, FFT_SIZE as i64);
    // println!("1D forward fft took {} millis", now.elapsed().as_millis());

    //af_print!("Output", output);



    // let array: Array<Complex<T>> = constant!(Complex::<T>::new(0.0, 2.0); FFT_SIZE, FFT_SIZE, FFT_SIZE);
    let now = Instant::now();
    const SIZE: usize = FFT_SIZE.pow(3) as usize;
    let values = vec![Complex::<T>::new(0.0, 2.0); SIZE];
    // for (i, v) in (0..).zip(&mut values) {
    //     *v = Complex::<T>::new(i as T,-1.0*i as T);
    // }
    let dims = Dim4::new(&[FFT_SIZE, FFT_SIZE, FFT_SIZE, 1]);
    let mut array = Array::new(&values, dims);
    println!("3D init array took {} millis", now.elapsed().as_millis());
    //af_print!("Output", array);
    let now = Instant::now();
    for _ in 0..10000 {
        fft3_inplace(&mut array, 1.0/(FFT_SIZE as f64).powf(3.0/2.0));
        //af_print!("Output", array);
        ifft3_inplace(&mut array, 1.0/(FFT_SIZE as f64).powf(3.0/2.0));
        //af_print!("Output", array);
    }
    println!("3D forward/inverse fft took {} millis", now.elapsed().as_millis());



}
