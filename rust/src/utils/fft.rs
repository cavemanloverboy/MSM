use arrayfire::*;
use num::{Complex, Float, FromPrimitive};
use crate::utils::{
    grid::Dimensions
};
use anyhow::Result;

pub fn forward<T>(
    array: &Array<Complex<T>>,
    dims: Dimensions,
    size: usize,
) -> Result<Array<Complex<T>>>
where
    T: Float + FloatingPoint,
    Complex<T>: HasAfEnum + FloatingPoint + HasAfEnum<ComplexOutType = Complex<T>>,
    <Complex<T> as arrayfire::HasAfEnum>::ComplexOutType: HasAfEnum
{
    // Compute dimension specific normalization factor
    let norm_factor: f64 = 1.0 / (size as f64).powf(dims as u8 as f64 / 2.0);

    // Handle ffts for different dimensions
    match dims {
        Dimensions::One => Ok(fft(array, norm_factor, size as i64)),
        Dimensions::Two => Ok(fft2(array, norm_factor, size as i64, size as i64)),
        Dimensions::Three => Ok(fft3(array, norm_factor, size as i64, size as i64, size as i64)),
    }
}

pub fn inverse<T>(
    array: &Array<Complex<T>>,
    dims: Dimensions,
    size: usize,
)-> Result<Array<Complex<T>>>
where
    T: Float + FloatingPoint,
    Complex<T>: HasAfEnum + FloatingPoint + HasAfEnum<ComplexOutType = Complex<T>>,
    <Complex<T> as arrayfire::HasAfEnum>::ComplexOutType: HasAfEnum
{
    // Compute dimension specific normalization factor
    let norm_factor: f64 = 1.0 / (size as f64).powf(dims as u8 as f64 / 2.0);

    // Handle ffts for different dimensions
    match dims {
        Dimensions::One => Ok(ifft(array, norm_factor, size as i64)),
        Dimensions::Two => Ok(ifft2(array, norm_factor, size as i64, size as i64)),
        Dimensions::Three => Ok(ifft3(array, norm_factor, size as i64, size as i64, size as i64)),
    }
}

pub fn forward_inplace<T>(
    array: &mut Array<Complex<T>>,
    dims: Dimensions,
    size: usize,
) -> Result<()>
where
    T: Float,
    Complex<T>: HasAfEnum + ComplexFloating,
{
    // Compute dimension specific normalization factor
    let norm_factor: f64 = 1.0 / (size as f64).powf(dims as u8 as f64 / 2.0);

    // Handle ffts for different dimensions
    match dims {
        Dimensions::One => Ok(fft_inplace(array, norm_factor)),
        Dimensions::Two => Ok(fft2_inplace(array, norm_factor)),
        Dimensions::Three => Ok(fft3_inplace(array, norm_factor)),
    }
}

pub fn inverse_inplace<T>(
    array: &mut Array<Complex<T>>,
    dims: Dimensions,
    size: usize,
)-> Result<()>
where
    T: Float,
    Complex<T>: HasAfEnum + ComplexFloating,
{
    // Compute dimension specific normalization factor
    let norm_factor: f64 = 1.0 / (size as f64).powf(dims as u8 as f64 / 2.0);

    // Handle ffts for different dimensions
    match dims {
        Dimensions::One => Ok(ifft_inplace(array, norm_factor)),
        Dimensions::Two => Ok(ifft2_inplace(array, norm_factor)),
        Dimensions::Three => Ok(ifft3_inplace(array, norm_factor)),
    }
}


pub fn get_kgrid<T>(
    dx: T,
    size: usize,
) -> Vec<T>
where
    T: Float + FromPrimitive,
{
    // Ensure grid is odd
    assert!(size % 2 == 0);

    // Initialize kgrid
    let mut kgrid = vec![T::zero(); size];

    for (k, mut i) in kgrid.iter_mut().zip(0..size as i64) {

        if i < (size as i64 / 2) {
            *k = T::from_i64(i).unwrap() / (T::from_usize(size).unwrap() * dx);
        } else {
            i -= size as i64;
            *k = T::from_i64(i).unwrap() / (T::from_usize(size).unwrap() * dx);
        }
    }

    kgrid
}

/// This computes `k2 = sum(k_i^2)` on the grid
pub fn spec_grid<T>(
    dx: T,
    dims: Dimensions,
    size: usize,
) -> Array<T>
where
    T: HasAfEnum + Float + FromPrimitive + ConstGenerator<OutType = T>
{
    // Get kgrid and square
    let kgrid_squared: Vec<T> = get_kgrid::<T>(dx, size)
        .iter()
        .map(|x| *x * *x)
        .collect();


    // Construct Array full of zeros
    let values = vec![T::zero(); size.pow(dims as u32)];
    let shape = match dims {
        Dimensions::One => (size as u64, 1, 1, 1),
        Dimensions::Two => (size as u64, size as u64, 1, 1),
        Dimensions::Three => (size as u64, size as u64, size as u64, 1),
    };
    let dim4 = Dim4::new(&[shape.0, shape.1, shape.2, shape.3]);
    let mut array = Array::new(&values, dim4);

    // Sum(k_i^2)
    for i in 0..dims as usize {
        
        // Shape of broadcasting array
        let mut bcast_shape = [1, 1, 1, 1];
        bcast_shape[i] = size as u64;
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

    mul(
        &array,
        &T::from_f64(2.0 * std::f64::consts::PI).unwrap().powf(T::from_f64(2.0).unwrap()),
        true
    )
}






#[test]
fn test_k_grid() {
    // Generate simple k grid and ensure it's correct
    let k_grid = get_kgrid::<f32>(0.25, 4);
    assert_eq!(k_grid, [0.0, 1.0, -2.0, -1.0]);

    let k_grid = get_kgrid::<f32>(30.0/256.0, 256);
    println!("max is {}", k_grid.iter().fold(0.0, |acc, &x| acc.max(x)));
}

#[test]
#[cfg(feature = "c64")]
fn test_k_grid_double() {
    // Generate simple k grid and ensure it's correct
    let k_grid = get_kgrid::<f64>(0.25, 4);
    assert_eq!(k_grid, [0.0, 1.0, -2.0, -1.0]);

    let k_grid = get_kgrid::<f64>(30.0/256.0, 256);
    println!("max is {}", k_grid.iter().fold(0.0, |acc, &x| acc.max(x)));
}

#[test]
fn test_spec_grid() {
    let size: usize = 4;
    let dims = Dimensions::Three;
    // Generate simple k grid and ensure it's correct
    let k_grid = get_kgrid::<f32>(0.25, 4);
    let spec_grid = spec_grid::<f32>(0.25, dims, size);

    // Manually build array
    let mut values = vec![0.0; size.pow(dims as u32)];
    for i in 0..size {
        for j in 0..size{
            for k in 0..size {
                let q = i + j*size + k*size*size;
                values[q] = (k_grid[i]*k_grid[i] + k_grid[j]*k_grid[j] + k_grid[k]*k_grid[k]) * ( 2.0 * std::f32::consts::PI ).powf(2.0);
            }
        }
    }
    //let array = Array::new(&values, Dim4::new(&[S as u64, S as u64, S as u64, 1]));

    let mut host = vec![0.0_f32; size.pow(dims as u32)];
    spec_grid.host(&mut host);
    for i in 0..values.len() {
        assert_eq!(values[i], host[i], "element {i} was not equal: {} != {}", values[i], host[i])
    }
}

#[test]
#[cfg(feature = "c64")]
fn test_spec_grid_double() {
    let size: usize = 4;
    let dims = Dimensions::Three;
    // Generate simple k grid and ensure it's correct
    let k_grid = get_kgrid::<f64>(0.25, 4);
    let spec_grid = spec_grid::<f64>(0.25, dims, size);

    // Manually build array
    let mut values = vec![0.0; size.pow(dims as u32)];
    for i in 0..size {
        for j in 0..size{
            for k in 0..size {
                let q = i + j*size + k*size*size;
                values[q] = (k_grid[i]*k_grid[i] + k_grid[j]*k_grid[j] + k_grid[k]*k_grid[k]) * ( 2.0 * std::f64::consts::PI ).powf(2.0);
            }
        }
    }
    //let array = Array::new(&values, Dim4::new(&[S as u64, S as u64, S as u64, 1]));

    let mut host = vec![0.0_f64; size.pow(dims as u32)];
    spec_grid.host(&mut host);
    for i in 0..values.len() {
        assert_eq!(values[i], host[i], "element {i} was not equal: {} != {}", values[i], host[i])
    }
}