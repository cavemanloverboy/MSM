
#[test]
fn test_arrayfire_1_d_inplace_c32_integration() {
    
    // Gather requirements for unit test
    use arrayfire::{Dim4, Array};
    use msm::utils::{
        fft::{forward_inplace, inverse_inplace},
        grid::check_norm
    };
    use approx::assert_abs_diff_eq;
    use num::Complex;

    type T = f32;

    // Set dimension, fft size
    const K: usize = 1;
    const S: usize = 8;
    const SIZE: usize = S.pow(K as u32);
    let length: T = 128.0;

    // c32 values to go in array
    let values = [Complex::<T>::new(0.0, length.powf(-1.0/2.0)); SIZE];
    

    // Specify dimensions of
    let dims = Dim4::new(&[S as u64, 1, 1, 1]);

    // Initialize array
    let mut array : Array<Complex<T>> = Array::new(&values, dims);
    let dx = length/S as T; // This is manually calculated to make array norm = 1
    //arrayfire::af_print!("array", &array);
    debug_assert!(check_norm::<T>(&array, dx, num::FromPrimitive::from_usize(K).unwrap()));

    // Perform inplace FFT + inverse, ensuring normalization in kspace
    forward_inplace::<T>(&mut array, num::FromPrimitive::from_usize(K).unwrap(), S).expect("forward fft failed");
    //arrayfire::af_print!("array", &array);
    let dk = dx;//1.0/length*(S as T).sqrt();
    debug_assert!(check_norm::<T>(&array, dk, num::FromPrimitive::from_usize(K).unwrap())); 
    inverse_inplace::<T>(&mut array, num::FromPrimitive::from_usize(K).unwrap(), S).expect("inverse fft failed");

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
    forward_inplace::<T>(&mut array, num::FromPrimitive::from_usize(K).unwrap(), S).expect("forward fft failed");
    inverse_inplace::<T>(&mut array, num::FromPrimitive::from_usize(K).unwrap(), S).expect("inverse fft failed");

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
    use msm::utils::{
        fft::{forward_inplace, inverse_inplace},
        grid::check_norm
    };
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
    let dx = 1.0/8.0; // This is manually calculated to make array norm = 1
    //debug_assert!(check_norm::<T>(&array, d, num::FromPrimitive::from_usize(K).unwrap()));

    // Perform inplace FFT + inverse
    forward_inplace::<T>(&mut array, num::FromPrimitive::from_usize(K).unwrap(), S).expect("forward fft failed");
    //arrayfire::af_print!("post fft", &array);
    //debug_assert!(check_norm::<T>(&array, 1.0/0.25/16.0)); // TODO: figure out why this is 1, num::FromPrimitive::from_usize(K).unwrap()...
    inverse_inplace::<T>(&mut array, num::FromPrimitive::from_usize(K).unwrap(), S).expect("inverse fft failed");

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
    forward_inplace::<T>(&mut array, num::FromPrimitive::from_usize(K).unwrap(), S).expect("forward fft failed");
    inverse_inplace::<T>(&mut array, num::FromPrimitive::from_usize(K).unwrap(), S).expect("inverse fft failed");

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
    forward_inplace::<T>(&mut array, num::FromPrimitive::from_usize(K).unwrap(), S).expect("forward fft failed");
    //af_print!("fwd", array);
    inverse_inplace::<T>(&mut array, num::FromPrimitive::from_usize(K).unwrap(), S).expect("inverse fft failed");
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
    forward_inplace::<T>(&mut array, num::FromPrimitive::from_usize(K).unwrap(), S).expect("forward fft failed");
    inverse_inplace::<T>(&mut array, num::FromPrimitive::from_usize(K).unwrap(), S).expect("inverse fft failed");

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
    let output = forward::<T>(& array, num::FromPrimitive::from_usize(K).unwrap(), S).expect("forward fft failed");
    let output = inverse::<T>(& output, num::FromPrimitive::from_usize(K).unwrap(), S).expect("inverse fft failed");

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
    let output = forward::<T>(& array, num::FromPrimitive::from_usize(K).unwrap(), S).expect("forward fft failed");
    let output = inverse::<T>(& output, num::FromPrimitive::from_usize(K).unwrap(), S).expect("inverse fft failed");

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
    let output = forward::<T>(& array, num::FromPrimitive::from_usize(K).unwrap(), S).expect("forward fft failed");
    let output = inverse::<T>(& output, num::FromPrimitive::from_usize(K).unwrap(), S).expect("inverse fft failed");

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
    let output = forward::<T>(& array, num::FromPrimitive::from_usize(K).unwrap(), S).expect("forward fft failed");
    let output = inverse::<T>(& output, num::FromPrimitive::from_usize(K).unwrap(), S).expect("inverse fft failed");

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
    let output = forward::<T>(& array, num::FromPrimitive::from_usize(K).unwrap(), S).expect("forward fft failed");
    //af_print!("fwd", array);
    let output = inverse::<T>(& output, num::FromPrimitive::from_usize(K).unwrap(), S).expect("inverse fft failed");
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
    let output = forward::<T>(& array, num::FromPrimitive::from_usize(K).unwrap(), S).expect("forward fft failed");
    let output = inverse::<T>(& output, num::FromPrimitive::from_usize(K).unwrap(), S).expect("inverse fft failed");

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
