
use arrayfire::{Array, FloatingPoint, HasAfEnum, sum_all, conjg, mul, isnan, isinf, Fromf64, ComplexFloating};
use num::{Float, Complex, FromPrimitive, ToPrimitive};
use std::fmt::Display;
use approx::{assert_abs_diff_eq};
use crate::utils::error::MSMError;


pub fn normalize<T, const K: usize>(
    grid: &mut Array<Complex<T>>,
    dx: T,
)
where
    T: Float + FloatingPoint + FromPrimitive + Display + Fromf64,
    Complex<T>: HasAfEnum + ComplexFloating + FloatingPoint + Default + HasAfEnum<ComplexOutType = Complex<T>> + HasAfEnum<AggregateOutType = Complex<T>> +  HasAfEnum<AbsOutType = T> + HasAfEnum<BaseType = T>,
{
    
    // Compute norm as grid * conjg(grid)
    let norm = sum_all(
        &mul(
            grid,
            &conjg(grid),
            false
        )
    );
    let norm = vec![Complex::<T>::new((dx.powf(-T::from_usize(K).unwrap())/norm.0).sqrt(), T::zero()); grid.elements()];
    let norm = Array::new(&norm, grid.dims());
    *grid = mul(grid, &norm, false);
}

pub fn check_norm<T, const K: usize>(
    grid: &Array<Complex<T>>,
    dx: T,
) -> bool
where
    T: Float + FloatingPoint + FromPrimitive + Display + Fromf64 + ToPrimitive,
    Complex<T>: HasAfEnum + ComplexFloating + FloatingPoint + Default + HasAfEnum<ComplexOutType = Complex<T>> + HasAfEnum<AggregateOutType = Complex<T>> +  HasAfEnum<AbsOutType = T> + HasAfEnum<BaseType = T>,
{
    
    // Compute norm as grid * conjg(grid)
    let norm = sum_all(
        &mul(
            grid,
            &conjg(grid),
            false
        )
    );
    

    assert_abs_diff_eq!(
        T::to_f64(
            &(
                norm.0 * dx.powf(T::from_usize(K).unwrap())
            )
        ).unwrap(),
        1.0,
        epsilon = 1e-6);
    (norm.0*dx.powf(T::from_usize(K).unwrap()) - T::one()).abs() < T::from_f64(1e-6).unwrap()
}



pub fn check_complex_for_nans<T>(
    array: &Array<Complex<T>>
)-> bool
where
    T: Float + HasAfEnum + Fromf64 + std::fmt::Display,
    Complex<T>: HasAfEnum + HasAfEnum<AggregateOutType = Complex<T>> + HasAfEnum<BaseType = T>,
{
    // check for nans
    let check: Array<bool> = isnan(&array);
    let nan_sum = sum_all(&check);
    let is_nan = (nan_sum.0 + nan_sum.1) > 0;

    // check for infs
    let check: Array<bool> = isinf(&array);
    let inf_sum = sum_all(&check);
    let is_inf = (inf_sum.0 + inf_sum.0) > 0;

    let is_bad = is_nan || is_inf;


    //println!("continuing {} with {} nans and {} infs", !is_bad, nan_sum.0 + nan_sum.1, inf_sum.0 + inf_sum.0);
    !is_bad
}

pub fn check_for_nans<T>(
    array: &Array<T>
)-> bool
where
    T: Float + HasAfEnum + Fromf64 + HasAfEnum<AggregateOutType = T> + HasAfEnum<BaseType = T>,
{
    // check for nans
    let check: Array<bool> = isnan(&array);
    let nan_sum = sum_all(&check);
    let is_nan = (nan_sum.0 + nan_sum.1) > 0;

    // check for infs
    let check: Array<bool> = isinf(&array);
    let inf_sum = sum_all(&check);
    let is_inf = (inf_sum.0 + inf_sum.0) > 0;

    let is_bad = is_nan || is_inf;

    //println!("continuing {} with {} nans and {} infs", !is_bad, nan_sum.0 + nan_sum.1, inf_sum.0 + inf_sum.0);
    !is_bad
}


#[test]
fn test_normalize_arrayfire_array_1d() {

    use arrayfire::Dim4;

    // Define type, size
    type T = f32;
    const S: usize = 8;
    const K: usize = 1;
    let dx = 1.0 / S as T;

    // Define values that go into array
    let values = [Complex::<T>::new(1.0, 1.0); S.pow(K as u32)];

    // Define Array
    let dims = Dim4::new(&[S.pow(K as u32) as u64, 1, 1, 1]);
    let mut array = Array::new(&values, dims);


    normalize::<T, K>(&mut array, dx);
    //arrayfire::af_print!("normalized array", array);


    let norm_check = sum_all(
        &mul(
            &array,
            &conjg(&array),
            false
        )
    ).0 * dx.powf(K as T);

    assert_eq!(array.dims(), dims);
    assert_abs_diff_eq!(norm_check, 1.0, epsilon = T::from_f64(1e-6).unwrap());
    assert!(check_norm::<T, K>(&array, dx));
}


#[test]
fn test_normalize_arrayfire_array_2d() {

    use arrayfire::Dim4;

    // Define type, size
    type T = f32;
    const S: usize = 8;
    const K: usize = 2;
    let dx = 1.0 / S as T;

    // Define values that go into array
    let values = [Complex::<T>::new(1.0, 1.0); S.pow(K as u32)];

    // Define Array
    let dims = Dim4::new(&[S as u64, S as u64, 1, 1]);
    let mut array = Array::new(&values, dims);


    normalize::<T, K>(&mut array, dx);
    //arrayfire::af_print!("normalized array", array);


    let norm_check = sum_all(
        &mul(
            &array,
            &conjg(&array),
            false
        )
    ).0 * dx.powf(K as T);

    assert_eq!(array.dims(), dims);
    assert_abs_diff_eq!(norm_check, 1.0, epsilon = T::from_f64(1e-6).unwrap());
    assert!(check_norm::<T, K>(&array, dx));
}


#[test]
fn test_normalize_arrayfire_array_3d() {

    use arrayfire::Dim4;

    // Define type, size
    type T = f32;
    const S: usize = 8;
    const K: usize = 3;
    let dx = 1.0 / S as T;

    // Define values that go into array
    let values = [Complex::<T>::new(1.0, 1.0); S.pow(K as u32)];

    // Define Array
    let dims = Dim4::new(&[S as u64, S as u64, S as u64, 1]);
    let mut array = Array::new(&values, dims);


    normalize::<T, K>(&mut array, dx);
    //arrayfire::af_print!("normalized array", array);


    let norm_check = sum_all(
        &mul(
            &array,
            &conjg(&array),
            false
        )
    ).0 * dx.powf(K as T);
 
    assert_eq!(array.dims(), dims);
    assert_abs_diff_eq!(norm_check, 1.0, epsilon = T::from_f64(1e-6).unwrap());
    assert!(check_norm::<T, K>(&array, dx));
}




#[test]
#[cfg_attr(not(feature="c64"), ignore)]
fn test_normalize_arrayfire_array_1d_f64() {

    use arrayfire::Dim4;

    // Define type, size
    type T = f64;
    const S: usize = 8;
    const K: usize = 1;
    let dx = 1.0 / S as T;

    // Define values that go into array
    let values = [Complex::<T>::new(1.0, 1.0); S.pow(K as u32)];

    // Define Array
    let dims = Dim4::new(&[S.pow(K as u32) as u64, 1, 1, 1]);
    let mut array = Array::new(&values, dims);


    normalize::<T, K>(&mut array, dx);
    arrayfire::af_print!("normalized array", array);


    let norm_check = sum_all(
        &mul(
            &array,
            &conjg(&array),
            false
        )
    ).0 * dx.powf(K as T);

    assert_eq!(array.dims(), dims);
    assert_abs_diff_eq!(norm_check, 1.0, epsilon = T::from_f64(1e-6).unwrap());
    assert!(check_norm::<T, K>(&array, dx));
}


#[test]
#[cfg_attr(not(feature="c64"), ignore)]
fn test_normalize_arrayfire_array_2d_f64() {

    use arrayfire::Dim4;

    // Define type, size
    type T = f64;
    const S: usize = 8;
    const K: usize = 2;
    let dx = 1.0 / S as T;

    // Define values that go into array
    let values = [Complex::<T>::new(1.0, 1.0); S.pow(K as u32)];

    // Define Array
    let dims = Dim4::new(&[S as u64, S as u64, 1, 1]);
    let mut array = Array::new(&values, dims);


    normalize::<T, K>(&mut array, dx);
    arrayfire::af_print!("normalized array", array);


    let norm_check = sum_all(
        &mul(
            &array,
            &conjg(&array),
            false
        )
    ).0 * dx.powf(K as T);

    assert_eq!(array.dims(), dims);
    assert_abs_diff_eq!(norm_check, 1.0, epsilon = T::from_f64(1e-6).unwrap());
    assert!(check_norm::<T, K>(&array, dx));
}


#[test]
#[cfg_attr(not(feature="c64"), ignore)]
fn test_normalize_arrayfire_array_3d_f64() {

    use arrayfire::Dim4;

    // Define type, size
    type T = f64;
    const S: usize = 8;
    const K: usize = 3;
    let dx = 1.0 / S as T;

    // Define values that go into array
    let values = [Complex::<T>::new(1.0, 1.0); S.pow(K as u32)];

    // Define Array
    let dims = Dim4::new(&[S as u64, S as u64, S as u64, 1]);
    let mut array = Array::new(&values, dims);


    normalize::<T, K>(&mut array, dx);
    arrayfire::af_print!("normalized array", array);


    let norm_check = sum_all(
        &mul(
            &array,
            &conjg(&array),
            false
        )
    ).0 * dx.powf(K as T);
 
    assert_eq!(array.dims(), dims);
    assert_abs_diff_eq!(norm_check, 1.0, epsilon = T::from_f64(1e-6).unwrap());
    assert!(check_norm::<T, K>(&array, dx));
}


