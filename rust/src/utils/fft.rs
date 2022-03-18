use arrayfire::*;
use num::{Complex, Float, FromPrimitive};
use crate::utils::error::MSMError;

pub fn forward<T, const K: usize, const S: usize>(
    array: &Array<Complex<T>>
) -> Result<Array<Complex<T>>, MSMError>
where
    T: Float + FloatingPoint,
    Complex<T>: HasAfEnum + FloatingPoint + HasAfEnum<ComplexOutType = Complex<T>>,
    <Complex<T> as arrayfire::HasAfEnum>::ComplexOutType: HasAfEnum
{
    // Compute dimension specific normalization factor
    let norm_factor: f64 = 1.0 / (S as f64).powf(K as f64 / 2.0);

    // Handle ffts for different dimensions
    match K {
        1 => Ok( fft(array, norm_factor, S as i64)),
        2 => Ok(fft2(array, norm_factor, S as i64, S as i64)),
        3 => Ok(fft3(array, norm_factor, S as i64, S as i64, S as i64)),
        _ => Err(MSMError::InvalidNumDumensions(K))
    }
}

pub fn inverse<T, const K: usize, const S: usize>(
    array: &Array<Complex<T>>
)-> Result<Array<Complex<T>>, MSMError>
where
    T: Float + FloatingPoint,
    Complex<T>: HasAfEnum + FloatingPoint + HasAfEnum<ComplexOutType = Complex<T>>,
    <Complex<T> as arrayfire::HasAfEnum>::ComplexOutType: HasAfEnum
{
    // Compute dimension specific normalization factor
    let norm_factor: f64 = 1.0 / (S as f64).powf(K as f64 / 2.0);

    // Handle ffts for different dimensions
    match K {
        1 => Ok( ifft(array, norm_factor, S as i64)),
        2 => Ok(ifft2(array, norm_factor, S as i64, S as i64)),
        3 => Ok(ifft3(array, norm_factor, S as i64, S as i64, S as i64)),
        _ => Err(MSMError::InvalidNumDumensions(K))
    }
}

pub fn forward_inplace<T, const K: usize, const S: usize>(array: &mut Array<Complex<T>>) -> Result<(), MSMError>
where
    T: Float,
    Complex<T>: HasAfEnum + ComplexFloating,
{
    // Compute dimension specific normalization factor
    let norm_factor: f64 = 1.0 / (S as f64).powf(K as f64 / 2.0);

    // Handle ffts for different dimensions
    match K {
        1 => Ok( fft_inplace(array, norm_factor)),
        2 => Ok(fft2_inplace(array, norm_factor)),
        3 => Ok(fft3_inplace(array, norm_factor)),
        _ => Err(MSMError::InvalidNumDumensions(K))
    }
}

pub fn inverse_inplace<T, const K: usize, const S: usize>(array: &mut Array<Complex<T>>)-> Result<(), MSMError>
where
    T: Float,
    Complex<T>: HasAfEnum + ComplexFloating,
{
    // Compute dimension specific normalization factor
    let norm_factor: f64 = 1.0 / (S as f64).powf(K as f64 / 2.0);

    // Handle ffts for different dimensions
    match K {
        1 => Ok( ifft_inplace(array, norm_factor)),
        2 => Ok(ifft2_inplace(array, norm_factor)),
        3 => Ok(ifft3_inplace(array, norm_factor)),
        _ => Err(MSMError::InvalidNumDumensions(K))
    }
}


pub fn get_kgrid<T, const S: usize>(
    dx: T, 
) -> [T; S]
where
    T: Float + FromPrimitive,
{
    // Ensure grid is odd
    assert!(S % 2 == 0);

    // Initialize kgrid
    let mut kgrid = [T::zero(); S];

    for (k, mut i) in kgrid.iter_mut().zip(0..S as i64) {

        if i < (S as i64 / 2) {
            *k = T::from_i64(i).unwrap() / (T::from_usize(S).unwrap() * dx);
        } else {
            i -= S as i64;
            *k = T::from_i64(i).unwrap() / (T::from_usize(S).unwrap() * dx);
        }
    }

    kgrid
}

/// This computes `k2 = sum(k_i^2)` on the grid
pub fn spec_grid<T, const K: usize, const S: usize>(
    dx: T,
    shape: (u64, u64, u64, u64)
) -> Array<T>
where
    T: HasAfEnum + Float + FromPrimitive
{
    // Get kgrid and square
    let kgrid_squared: Vec<T> = get_kgrid::<T, S>(dx)
        .iter()
        .map(|x| *x * *x)
        .collect();


    // Construct Array full of zeros
    let values = vec![T::zero(); S.pow(K as u32)];
    let dims = Dim4::new(&[shape.0, shape.1, shape.2, shape.3]);
    let mut array = Array::new(&values, dims);

    // Sum(k_i^2)
    for i in 0..K {
        
        // Shape of broadcasting array
        let mut bcast_shape = [1, 1, 1, 1];
        bcast_shape[i] = S as u64;
        let bcast_dims = Dim4::new(&bcast_shape);

        // Construct brodcasting array
        let bcast_array = Array::new(&kgrid_squared, bcast_dims);

        // Add bcast_array to array
        array = add(
            &array,
            &bcast_array,
            true
        )
    }

    array
}






#[test]
fn test_k_grid() {

    // Generate simple k grid and ensure it's correct
    let k_grid = get_kgrid::<f32, 4>(0.25);
    assert_eq!(k_grid, [0.0, 1.0, -2.0, -1.0])
}