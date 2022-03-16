
#[test]
fn test_arrayfire_1_d_inplace_c32_integration() {
    
    // Gather requirements for unit test
    use arrayfire::{Dim4, Array};
    use msm::utils::fft::{forward_inplace, inverse_inplace};
    use approx::assert_abs_diff_eq;
    use num::Complex;

    type T = f32;

    // Set dimension, fft size
    const K: usize = 1;
    const S: usize = 2;
    const SIZE: usize = S.pow(K as u32);

    // c32 values to go in array
    let values = [Complex::<T>::new(0.0, 2.0); SIZE];

    // Specify dimensions of
    let dims = Dim4::new(&[S as u64, 1, 1, 1]);

    // Initialize array
    let mut array : Array<Complex<T>> = Array::new(&values, dims);

    // Perform inplace FFT + inverse
    forward_inplace::<T, K, S>(&mut array).expect("forward fft failed");
    inverse_inplace::<T, K, S>(&mut array).expect("inverse fft failed");

    // Create hosts for comparison
    let mut vec_array = vec![Complex::<T>::default(); SIZE];

    // Host
    array.host(&mut vec_array);

    // Check norm of difference is small
    assert_abs_diff_eq!(

        vec_array
            .iter()
            .zip(&values)
            .fold(0.0, |acc: T, (x,y)| acc + (x-y).norm()),

            0.0,

            epsilon = 1e-6
    );
}

#[test]
#[cfg_attr(not(feature = "c64"), ignore)]
fn test_arrayfire_1_d_inplace_c64_integration() {
    
    // Gather requirements for unit test
    use arrayfire::{Dim4, Array};
    use msm::utils::fft::{forward_inplace, inverse_inplace};
    use approx::assert_abs_diff_eq;
    use num::Complex;

    type T = f64;

    // Set dimension, fft size
    const K: usize = 1;
    const S: usize = 2;
    const SIZE: usize = S.pow(K as u32);

    // c64 values to go in array
    let values = [Complex::<T>::new(0.0, 2.0); SIZE];

    // Specify dimensions of
    let dims = Dim4::new(&[S as u64, 1, 1, 1]);

    // Initialize array
    let mut array = Array::new(&values, dims);

    // Perform inplace FFT + inverse
    forward_inplace::<T, K, S>(&mut array).expect("forward fft failed");
    inverse_inplace::<T, K, S>(&mut array).expect("inverse fft failed");

    // Create hosts for comparison
    let mut vec_array = vec![Complex::<T>::default(); SIZE];

    // Host
    array.host(&mut vec_array);

    // Check norm of diffence is small
    assert_abs_diff_eq!(

        vec_array
            .iter()
            .zip(&values)
            .fold(0.0, |acc: T, (x,y)| acc + (x-y).norm()),

            0.0,

            epsilon = 1e-6
    );
}

#[test]
fn test_arrayfire_2_d_inplace_c32_integration() {
    
    // Gather requirements for unit test
    use arrayfire::{Dim4, Array};
    use msm::utils::fft::{forward_inplace, inverse_inplace};
    use approx::assert_abs_diff_eq;
    use num::Complex;

    type T = f32;

    // Set dimension, fft size
    const K: usize = 2;
    const S: usize = 2;
    const SIZE: usize = S.pow(K as u32);

    // c32 values to go in array
    let values = [Complex::<T>::new(0.0, 2.0); SIZE];

    // Specify dimensions of
    let dims = Dim4::new(&[S as u64, S as u64, 1, 1]);

    // Initialize array
    let mut array = Array::new(&values, dims);

    // Perform inplace FFT + inverse
    forward_inplace::<T, K, S>(&mut array).expect("forward fft failed");
    inverse_inplace::<T, K, S>(&mut array).expect("inverse fft failed");

    // Create hosts for comparison
    let mut vec_array = vec![Complex::<T>::default(); SIZE];

    // Host
    array.host(&mut vec_array);

    // Check norm of diffence is small
    assert_abs_diff_eq!(

        vec_array
            .iter()
            .zip(&values)
            .fold(0.0, |acc: T, (x,y)| acc + (x-y).norm()),

            0.0,

            epsilon = 1e-6
    );  
}

#[test]
#[cfg_attr(not(feature = "c64"), ignore)]
fn test_arrayfire_2_d_inplace_c64_integration() {
    
    // Gather requirements for unit test
    use arrayfire::{Dim4, Array};
    use msm::utils::fft::{forward_inplace, inverse_inplace};
    use approx::assert_abs_diff_eq;
    use num::Complex;

    type T = f64;

    // Set dimension, fft size
    const K: usize = 2;
    const S: usize = 2;
    const SIZE: usize = S.pow(K as u32);

    // c64 values to go in array
    let values = [Complex::<T>::new(0.0, 2.0); SIZE];

    // Specify dimensions of
    let dims = Dim4::new(&[S as u64, S as u64, 1, 1]);

    // Initialize array
    let mut array = Array::new(&values, dims);

    // Perform inplace FFT + inverse
    forward_inplace::<T, K, S>(&mut array).expect("forward fft failed");
    inverse_inplace::<T, K, S>(&mut array).expect("inverse fft failed");

    // Create hosts for comparison
    let mut vec_array = vec![Complex::<T>::default(); SIZE];

    // Host
    array.host(&mut vec_array);

    // Check norm of diffence is small
    assert_abs_diff_eq!(

        vec_array
            .iter()
            .zip(&values)
            .fold(0.0, |acc: T, (x,y)| acc + (x-y).norm()),

            0.0,

            epsilon = 1e-6
    );
}

#[test]
fn test_arrayfire_3_d_inplace_c32_integration() {
    
    // Gather requirements for unit test
    use arrayfire::{Dim4, Array};
    use msm::utils::fft::{forward_inplace, inverse_inplace};
    use approx::assert_abs_diff_eq;
    use num::Complex;

    type T = f32;

    // Set dimension, fft size
    const K: usize = 3;
    const S: usize = 2;
    const SIZE: usize = S.pow(K as u32);

    // c32 values to go in array
    let values = [Complex::<T>::new(0.0, 2.0); SIZE];

    // Specify dimensions of
    let dims = Dim4::new(&[S as u64, S as u64, S as u64, 1]);

    // Initialize array
    let mut array = Array::new(&values, dims);

    // Perform inplace FFT + inverse
    //af_print!("orig", array);
    forward_inplace::<T, K, S>(&mut array).expect("forward fft failed");
    //af_print!("fwd", array);
    inverse_inplace::<T, K, S>(&mut array).expect("inverse fft failed");
    //af_print!("inv", array);

    // Create hosts for comparison
    let mut vec_array = vec![Complex::<T>::default(); SIZE];

    // Host
    array.host(&mut vec_array);
    
    println!("{:?}", vec_array);
    println!("{:?}", values);

    // Check norm of diffence is small
    assert_abs_diff_eq!(

        vec_array
            .iter()
            .zip(&values)
            .fold(0.0, |acc: T, (x,y)| acc + (x-y).norm()),

            0.0,

            epsilon = 1e-6
    );
}

#[test]
#[cfg_attr(not(feature = "c64"), ignore)]
fn test_arrayfire_3_d_inplace_c64_integration() {
    
    // Gather requirements for unit test
    use arrayfire::{Dim4, Array};
    use msm::utils::fft::{forward_inplace, inverse_inplace};
    use approx::assert_abs_diff_eq;
    use num::Complex;

    type T = f64;

    // Set dimension, fft size
    const K: usize = 3;
    const S: usize = 2;
    const SIZE: usize = S.pow(K as u32);

    // c64 values to go in array
    let values = [Complex::<T>::new(0.0, 2.0); SIZE];

    // Specify dimensions of
    let dims = Dim4::new(&[S as u64, S as u64, S as u64, 1]);

    // Initialize array
    let mut array = Array::new(&values, dims);

    // Perform inplace FFT + inverse
    forward_inplace::<T, K, S>(&mut array).expect("forward fft failed");
    inverse_inplace::<T, K, S>(&mut array).expect("inverse fft failed");

    // Create hosts for comparison
    let mut vec_array = vec![Complex::<T>::default(); SIZE];


    // Host
    array.host(&mut vec_array);

    // Check norm of diffence is small
    assert_abs_diff_eq!(

        vec_array
            .iter()
            .zip(&values)
            .fold(0.0, |acc: T, (x,y)| acc + (x-y).norm()),

            0.0,

            epsilon = 1e-6
    );
}



#[test]
fn test_arrayfire_1_d_c32_integration() {
    
    // Gather requirements for unit test
    use arrayfire::{Dim4, Array};
    use msm::utils::fft::{forward, inverse};
    use approx::assert_abs_diff_eq;
    use num::Complex;

    type T = f32;

    // Set dimension, fft size
    const K: usize = 1;
    const S: usize = 2;
    const SIZE: usize = S.pow(K as u32);

    // c32 values to go in array
    let values = [Complex::<T>::new(0.0, 2.0); SIZE];

    // Specify dimensions of
    let dims = Dim4::new(&[S as u64, 1, 1, 1]);

    // Initialize array
    let array : Array<Complex<T>> = Array::new(&values, dims);

    // Perform inplace FFT + inverse
    let output = forward::<T, K, S>(& array).expect("forward fft failed");
    let output = inverse::<T, K, S>(& output).expect("inverse fft failed");

    // Create hosts for comparison
    let mut vec_array = vec![Complex::<T>::default(); SIZE];

    // Host
    output.host(&mut vec_array);

    // Check norm of difference is small
    assert_abs_diff_eq!(

        vec_array
            .iter()
            .zip(&values)
            .fold(0.0, |acc: T, (x,y)| acc + (x-y).norm()),

            0.0,

            epsilon = 1e-6
    );
}

#[test]
#[cfg_attr(not(feature = "c64"), ignore)]
fn test_arrayfire_1_d_c64_integration() {
    
    // Gather requirements for unit test
    use arrayfire::{Dim4, Array};
    use msm::utils::fft::{forward, inverse};
    use approx::assert_abs_diff_eq;
    use num::Complex;

    type T = f64;

    // Set dimension, fft size
    const K: usize = 1;
    const S: usize = 2;
    const SIZE: usize = S.pow(K as u32);

    // c64 values to go in array
    let values = [Complex::<T>::new(0.0, 2.0); SIZE];

    // Specify dimensions of
    let dims = Dim4::new(&[S as u64, 1, 1, 1]);

    // Initialize array
    let array = Array::new(&values, dims);

    // Perform inplace FFT + inverse
    let output = forward::<T, K, S>(& array).expect("forward fft failed");
    let output = inverse::<T, K, S>(& output).expect("inverse fft failed");

    // Create hosts for comparison
    let mut vec_array = vec![Complex::<T>::default(); SIZE];

    // Host
    output.host(&mut vec_array);

    // Check norm of diffence is small
    assert_abs_diff_eq!(

        vec_array
            .iter()
            .zip(&values)
            .fold(0.0, |acc: T, (x,y)| acc + (x-y).norm()),

            0.0,

            epsilon = 1e-6
    );
}

#[test]
fn test_arrayfire_2_d_c32_integration() {
    
    // Gather requirements for unit test
    use arrayfire::{Dim4, Array};
    use msm::utils::fft::{forward, inverse};
    use approx::assert_abs_diff_eq;
    use num::Complex;

    type T = f32;

    // Set dimension, fft size
    const K: usize = 2;
    const S: usize = 2;
    const SIZE: usize = S.pow(K as u32);

    // c32 values to go in array
    let values = [Complex::<T>::new(0.0, 2.0); SIZE];

    // Specify dimensions of
    let dims = Dim4::new(&[S as u64, S as u64, 1, 1]);

    // Initialize array
    let array = Array::new(&values, dims);

    // Perform inplace FFT + inverse
    let output = forward::<T, K, S>(& array).expect("forward fft failed");
    let output = inverse::<T, K, S>(& output).expect("inverse fft failed");

    // Create hosts for comparison
    let mut vec_array = vec![Complex::<T>::default(); SIZE];

    // Host
    output.host(&mut vec_array);

    // Check norm of diffence is small
    assert_abs_diff_eq!(

        vec_array
            .iter()
            .zip(&values)
            .fold(0.0, |acc: T, (x,y)| acc + (x-y).norm()),

            0.0,

            epsilon = 1e-6
    );  
}

#[test]
#[cfg_attr(not(feature = "c64"), ignore)]
fn test_arrayfire_2_d_c64_integration() {
    
    // Gather requirements for unit test
    use arrayfire::{Dim4, Array};
    use msm::utils::fft::{forward, inverse};
    use approx::assert_abs_diff_eq;
    use num::Complex;

    type T = f64;

    // Set dimension, fft size
    const K: usize = 2;
    const S: usize = 2;
    const SIZE: usize = S.pow(K as u32);

    // c64 values to go in array
    let values = [Complex::<T>::new(0.0, 2.0); SIZE];

    // Specify dimensions of
    let dims = Dim4::new(&[S as u64, S as u64, 1, 1]);

    // Initialize array
    let array = Array::new(&values, dims);

    // Perform inplace FFT + inverse
    let output = forward::<T, K, S>(& array).expect("forward fft failed");
    let output = inverse::<T, K, S>(& output).expect("inverse fft failed");

    // Create hosts for comparison
    let mut vec_array = vec![Complex::<T>::default(); SIZE];

    // Host
    output.host(&mut vec_array);

    // Check norm of diffence is small
    assert_abs_diff_eq!(

        vec_array
            .iter()
            .zip(&values)
            .fold(0.0, |acc: T, (x,y)| acc + (x-y).norm()),

            0.0,

            epsilon = 1e-6
    );
}

#[test]
fn test_arrayfire_3_d_c32_integration() {
    
    // Gather requirements for unit test
    use arrayfire::{Dim4, Array};
    use msm::utils::fft::{forward, inverse};
    use approx::assert_abs_diff_eq;
    use num::Complex;

    type T = f32;

    // Set dimension, fft size
    const K: usize = 3;
    const S: usize = 2;
    const SIZE: usize = S.pow(K as u32);

    // c32 values to go in array
    let values = [Complex::<T>::new(0.0, 2.0); SIZE];

    // Specify dimensions of
    let dims = Dim4::new(&[S as u64, S as u64, S as u64, 1]);

    // Initialize array
    let array = Array::new(&values, dims);

    // Perform inplace FFT + inverse
    //af_print!("orig", array);
    let output = forward::<T, K, S>(& array).expect("forward fft failed");
    //af_print!("fwd", array);
    let output = inverse::<T, K, S>(& output).expect("inverse fft failed");
    //af_print!("inv", array);

    // Create hosts for comparison
    let mut vec_array = vec![Complex::<T>::default(); SIZE];

    // Host
    output.host(&mut vec_array);
    
    //println!("{:?}", vec_array);
    //println!("{:?}", values);

    // Check norm of diffence is small
    assert_abs_diff_eq!(

        vec_array
            .iter()
            .zip(&values)
            .fold(0.0, |acc: T, (x,y)| acc + (x-y).norm()),

            0.0,

            epsilon = 1e-6
    );
}

#[test]
#[cfg_attr(not(feature = "c64"), ignore)]
fn test_arrayfire_3_d_c64_integration() {
    
    // Gather requirements for unit test
    use arrayfire::{Dim4, Array};
    use msm::utils::fft::{forward, inverse};
    use approx::assert_abs_diff_eq;
    use num::Complex;

    type T = f64;

    // Set dimension, fft size
    const K: usize = 3;
    const S: usize = 2;
    const SIZE: usize = S.pow(K as u32);

    // c64 values to go in array
    let values = [Complex::<T>::new(0.0, 2.0); SIZE];

    // Specify dimensions of
    let dims = Dim4::new(&[S as u64, S as u64, S as u64, 1]);

    // Initialize array
    let array = Array::new(&values, dims);

    // Perform inplace FFT + inverse
    let output = forward::<T, K, S>(& array).expect("forward fft failed");
    let output = inverse::<T, K, S>(& output).expect("inverse fft failed");

    // Create hosts for comparison
    let mut vec_array = vec![Complex::<T>::default(); SIZE];


    // Host
    output.host(&mut vec_array);

    // Check norm of diffence is small
    assert_abs_diff_eq!(

        vec_array
            .iter()
            .zip(&values)
            .fold(0.0, |acc: T, (x,y)| acc + (x-y).norm()),

            0.0,

            epsilon = 1e-6
    );
}
