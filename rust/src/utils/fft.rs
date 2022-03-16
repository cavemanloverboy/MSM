use arrayfire::*;
use num::{Complex, Float};

pub fn forward<T, const K: usize, const S: usize>(
    array: &Array<Complex<T>>
) -> Result<Array<<Complex<T> as arrayfire::HasAfEnum>::ComplexOutType>, MSMError>
where
    T: Float + FloatingPoint,
    Complex<T>: HasAfEnum + FloatingPoint,
    <Complex<T> as arrayfire::HasAfEnum>::ComplexOutType: HasAfEnum
{
    // Compute dimension specific normalization factor
    let norm_factor: f64 = 1.0 / (S as f64).powf(K as f64 / 2.0);

    // Handle ffts for different dimensions
    match K {
        1 => Ok( fft(array, norm_factor, S as i64)),
        2 => Ok(fft2(array, norm_factor, S as i64, S as i64)),
        3 => Ok(fft3(array, norm_factor, S as i64, S as i64, S as i64)),
        _ => Err(MSMError::InvalidNumDumensions(S))
    }
}

pub fn inverse<T, const K: usize, const S: usize>(
    array: &Array<Complex<T>>
)-> Result<Array<<Complex<T> as arrayfire::HasAfEnum>::ComplexOutType>, MSMError>
where
    T: Float + FloatingPoint,
    Complex<T>: HasAfEnum + FloatingPoint,
    <Complex<T> as arrayfire::HasAfEnum>::ComplexOutType: HasAfEnum
{
    // Compute dimension specific normalization factor
    let norm_factor: f64 = 1.0 / (S as f64).powf(K as f64 / 2.0);

    // Handle ffts for different dimensions
    match K {
        1 => Ok( ifft(array, norm_factor, S as i64)),
        2 => Ok(ifft2(array, norm_factor, S as i64, S as i64)),
        3 => Ok(ifft3(array, norm_factor, S as i64, S as i64, S as i64)),
        _ => Err(MSMError::InvalidNumDumensions(S))
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
        _ => Err(MSMError::InvalidNumDumensions(S))
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
        _ => Err(MSMError::InvalidNumDumensions(S))
    }
}


#[derive(Debug)]
pub enum MSMError {
    InvalidNumDumensions(usize),
}