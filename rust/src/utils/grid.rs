
use arrayfire::{Array, FloatingPoint, HasAfEnum, sum_all, conjg, mul, Fromf64, ComplexFloating};
use num::{Float, Complex, FromPrimitive, ToPrimitive};
use std::fmt::Display;
use approx::{assert_abs_diff_eq};


pub fn normalize<T, const K: usize>(
    grid: &mut Array<Complex<T>>,
    dx: T,
)
where
    T: Float + FloatingPoint + FromPrimitive + Display + Fromf64,
    Complex<T>: HasAfEnum + ComplexFloating + FloatingPoint + Default + HasAfEnum<ComplexOutType = Complex<T>> + HasAfEnum<AggregateOutType = Complex<T>> + HasAfEnum<BaseType = T>,
    <<num::Complex<T> as arrayfire::HasAfEnum>::AggregateOutType as arrayfire::HasAfEnum>::BaseType: Fromf64,
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
    Complex<T>: HasAfEnum + ComplexFloating + FloatingPoint + Default + HasAfEnum<ComplexOutType = Complex<T>> + HasAfEnum<AggregateOutType = Complex<T>> + HasAfEnum<BaseType = T>,
    <<num::Complex<T> as arrayfire::HasAfEnum>::AggregateOutType as arrayfire::HasAfEnum>::BaseType: Fromf64,
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