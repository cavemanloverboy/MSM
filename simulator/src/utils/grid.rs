use approx::assert_abs_diff_eq;
use arrayfire::{
    conjg, isinf, isnan, mul, real, sum_all, Array, ComplexFloating, FloatingPoint, Fromf64,
    HasAfEnum,
};
use num::{Complex, Float, FromPrimitive, ToPrimitive};
use num_derive::{FromPrimitive, ToPrimitive};
use serde::{Deserialize, Serialize};
use std::fmt::Display;

pub fn normalize<T>(grid: &mut Array<Complex<T>>, dx: T, dims: Dimensions)
where
    T: Float + FloatingPoint + FromPrimitive + Display + Fromf64,
    Complex<T>: HasAfEnum
        + ComplexFloating
        + FloatingPoint
        + HasAfEnum<ComplexOutType = Complex<T>>
        + HasAfEnum<AggregateOutType = Complex<T>>
        + HasAfEnum<AbsOutType = T>
        + HasAfEnum<BaseType = T>,
{
    // Compute norm as grid * conjg(grid)
    let norm = sum_all(&mul(grid, &conjg(grid), false));
    let norm = vec![
        Complex::<T>::new(
            (dx.powf(-T::from_usize(dims as usize).unwrap()) / norm.0).sqrt(),
            T::zero()
        );
        grid.elements()
    ];
    let norm = Array::new(&norm, grid.dims());
    *grid = mul(grid, &norm, false);
}

pub fn check_norm<T>(grid: &Array<Complex<T>>, dx: T, dim: Dimensions) -> bool
where
    T: Float
        + FloatingPoint
        + FromPrimitive
        + Display
        + HasAfEnum<BaseType = T>
        + HasAfEnum<AggregateOutType = T>
        + Fromf64
        + ToPrimitive,
    Complex<T>: HasAfEnum
        + ComplexFloating
        + FloatingPoint
        + HasAfEnum<ComplexOutType = Complex<T>>
        + HasAfEnum<AggregateOutType = Complex<T>>
        + HasAfEnum<AbsOutType = T>
        + HasAfEnum<BaseType = T>,
{
    // Compute norm as grid * conjg(grid)
    let norm: (T, T) = sum_all(&real(&mul(grid, &conjg(grid), false)));

    assert_abs_diff_eq!(
        T::to_f64(&(norm.0 * dx.powf(T::from_usize(dim as usize).unwrap()))).unwrap(),
        1.0,
        epsilon = 1e-4
    );

    (norm.0 * dx.powf(T::from_usize(dim as usize).unwrap()) - T::one()).abs()
        < T::from_f64(1e-4).unwrap()
}

pub fn check_complex_for_nans<T>(array: &Array<Complex<T>>) -> bool
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

pub fn check_for_nans<T>(array: &Array<T>) -> bool
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
    const D: Dimensions = Dimensions::One;
    let dx = 1.0 / S as T;

    // Define values that go into array
    let values = [Complex::<T>::new(1.0, 1.0); S.pow(D as u32)];

    // Define Array
    let dims = Dim4::new(&[S.pow(D as u32) as u64, 1, 1, 1]);
    let mut array = Array::new(&values, dims);

    normalize::<T>(&mut array, dx, D);
    //arrayfire::af_print!("normalized array", array);

    let norm_check = sum_all(&mul(&array, &conjg(&array), false)).0 * dx.powf(D as u8 as T);

    assert_eq!(array.dims(), dims);
    assert_abs_diff_eq!(norm_check, 1.0, epsilon = T::from_f64(1e-6).unwrap());
    assert!(check_norm::<T>(&array, dx, D));
}

#[test]
fn test_normalize_arrayfire_array_2d() {
    use arrayfire::Dim4;

    // Define type, size
    type T = f32;
    const S: usize = 8;
    const D: Dimensions = Dimensions::Two;
    let dx = 1.0 / S as T;

    // Define values that go into array
    let values = [Complex::<T>::new(1.0, 1.0); S.pow(D as u32)];

    // Define Array
    let dims = Dim4::new(&[S as u64, S as u64, 1, 1]);
    let mut array = Array::new(&values, dims);

    normalize::<T>(&mut array, dx, D);
    //arrayfire::af_print!("normalized array", array);

    let norm_check = sum_all(&mul(&array, &conjg(&array), false)).0 * dx.powf(D as u8 as T);

    assert_eq!(array.dims(), dims);
    assert_abs_diff_eq!(norm_check, 1.0, epsilon = T::from_f64(1e-6).unwrap());
    assert!(check_norm::<T>(&array, dx, D));
}

#[test]
fn test_normalize_arrayfire_array_3d() {
    use arrayfire::Dim4;

    // Define type, size
    type T = f32;
    const S: usize = 8;
    const D: Dimensions = Dimensions::Three;
    let dx = 1.0 / S as T;

    // Define values that go into array
    let values = [Complex::<T>::new(1.0, 1.0); S.pow(D as u32)];

    // Define Array
    let dims = Dim4::new(&[S as u64, S as u64, S as u64, 1]);
    let mut array = Array::new(&values, dims);

    normalize::<T>(&mut array, dx, D);
    //arrayfire::af_print!("normalized array", array);

    let norm_check = sum_all(&mul(&array, &conjg(&array), false)).0 * dx.powf(D as u8 as T);

    assert_eq!(array.dims(), dims);
    assert_abs_diff_eq!(norm_check, 1.0, epsilon = T::from_f64(1e-6).unwrap());
    assert!(check_norm::<T>(&array, dx, D));
}

#[test]
#[cfg_attr(not(feature = "c64"), ignore)]
fn test_normalize_arrayfire_array_1d_f64() {
    use arrayfire::Dim4;

    // Define type, size
    type T = f64;
    const S: usize = 8;
    const D: Dimensions = Dimensions::One;
    let dx = 1.0 / S as T;

    // Define values that go into array
    let values = [Complex::<T>::new(1.0, 1.0); S.pow(D as u32)];

    // Define Array
    let dims = Dim4::new(&[S.pow(D as u32) as u64, 1, 1, 1]);
    let mut array = Array::new(&values, dims);

    normalize::<T>(&mut array, dx, D);
    //arrayfire::af_print!("normalized array", array);

    let norm_check = sum_all(&mul(&array, &conjg(&array), false)).0 * dx.powf(D as u8 as T);

    assert_eq!(array.dims(), dims);
    assert_abs_diff_eq!(norm_check, 1.0, epsilon = T::from_f64(1e-6).unwrap());
    assert!(check_norm::<T>(&array, dx, D));
}

#[test]
#[cfg_attr(not(feature = "c64"), ignore)]
fn test_normalize_arrayfire_array_2d_f64() {
    use arrayfire::Dim4;

    // Define type, size
    type T = f64;
    const S: usize = 8;
    const D: Dimensions = Dimensions::Two;
    let dx = 1.0 / S as T;

    // Define values that go into array
    let values = [Complex::<T>::new(1.0, 1.0); S.pow(D as u32)];

    // Define Array
    let dims = Dim4::new(&[S as u64, S as u64, 1, 1]);
    let mut array = Array::new(&values, dims);

    normalize::<T>(&mut array, dx, D);
    //arrayfire::af_print!("normalized array", array);

    let norm_check = sum_all(&mul(&array, &conjg(&array), false)).0 * dx.powf(D as u8 as T);

    assert_eq!(array.dims(), dims);
    assert_abs_diff_eq!(norm_check, 1.0, epsilon = T::from_f64(1e-6).unwrap());
    assert!(check_norm::<T>(&array, dx, D));
}

#[test]
#[cfg_attr(not(feature = "c64"), ignore)]
fn test_normalize_arrayfire_array_3d_f64() {
    use arrayfire::Dim4;

    // Define type, size
    type T = f64;
    const S: usize = 8;
    const D: Dimensions = Dimensions::Three;
    let dx = 1.0 / S as T;

    // Define values that go into array
    let values = [Complex::<T>::new(1.0, 1.0); S.pow(D as u32)];

    // Define Array
    let dims = Dim4::new(&[S as u64, S as u64, S as u64, 1]);
    let mut array = Array::new(&values, dims);

    normalize::<T>(&mut array, dx, D);
    //arrayfire::af_print!("normalized array", array);

    let norm_check = sum_all(&mul(&array, &conjg(&array), false)).0 * dx.powf(D as u8 as T);

    assert_eq!(array.dims(), dims);
    assert_abs_diff_eq!(norm_check, 1.0, epsilon = T::from_f64(1e-6).unwrap());
    assert!(check_norm::<T>(&array, dx, D));
}

#[derive(
    Copy, Clone, Debug, Serialize, Deserialize, FromPrimitive, ToPrimitive, PartialEq, PartialOrd,
)]
pub enum Dimensions {
    One = 1,
    Two = 2,
    Three = 3,
}

/// This is used to make the parsing of elements when reading ICs much cleaner
pub trait IntoT<T> {
    fn into_t(self) -> Vec<T>;
}

impl<T> IntoT<T> for &[f64]
where
    T: FromPrimitive,
{
    fn into_t(self) -> Vec<T> {
        self.iter().map(|&x| T::from_f64(x).unwrap()).collect()
    }
}

#[test]
fn dimension_enum() {
    let _x: u8 = Dimensions::One as u8;
}
