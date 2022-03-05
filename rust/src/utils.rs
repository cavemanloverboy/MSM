use approx::assert_abs_diff_eq;
use conv::prelude::*;
use ndarray::parallel::prelude::*;
use ndarray::{
    arr1, arr2, arr3, array, s, Array, ArrayD, Axis, Dim, Dimension, IntoDimension, Ix1, Ix2, Ix3,
    IxDyn, ScalarOperand, Slice, SliceArg,
};
use num_traits::Float;
use rustfft::{num_complex::Complex, FftNum, FftPlanner};
use std::sync::Arc;
use std::thread;
use transpose::{transpose, transpose_inplace};

/// This struct is intended to be initialized at the
/// beginning of a simulation. It holds the forward
/// and inverse fft objects that perform FFTs. Here,
/// `T` specifies the float precision, `K` the dimensionality,
/// and `S` the FFT/grid size.
pub struct FftObject<T, const K: usize, const S: usize>
where
    T: FftNum + ValueFrom<usize> + ScalarOperand,
{
    // Planner
    pub planner: rustfft::FftPlanner<T>,
    // Forward operation
    pub fwd: std::sync::Arc<dyn rustfft::Fft<T>>,
    // Inverse operation
    pub inv: std::sync::Arc<dyn rustfft::Fft<T>>,
}

impl<T, const K: usize, const S: usize> FftObject<T, K, S>
where
    T: FftNum + ValueFrom<usize> + ScalarOperand,
{
    /// This `FftObject` constructor takes in a `size` (which is stored)
    /// and returns a struct containing the fft planner (`rustfft::FftPlanner<T>`)
    /// along with the associated forward and inverse operators.
    pub fn new() -> Self {
        // Create planner of type T
        let mut planner = FftPlanner::<T>::new();

        // Create forward and inverse plans
        let fwd = planner.plan_fft_forward(S);
        let inv = planner.plan_fft_inverse(S);

        // Pack into struct and return
        FftObject { planner, fwd, inv }
    }

    /// This function handles the forward FFT operation for Dimension = 1, 2, 3
    pub fn forward(&self, data: &mut Array<Complex<T>, Dim<[usize; K]>>) -> Result<(), MSMError>
    where
        T: FftNum + ValueFrom<usize> + ScalarOperand + Float + std::ops::AddAssign,
        Complex<T>: std::ops::DivAssign + std::ops::AddAssign + ScalarOperand,
        Dim<[usize; K]>: Dimension + IntoDimension,
    {
        // Ensure data provided is of the size supported
        assert_eq!(data.shape(), &[S; K]);

        // Forward needs division
        *data /= Complex {
            re: S.pow(K as u32).value_as::<T>().unwrap(),
            im: T::zero(),
        };

        // Dimension dependent forward
        match K {
            // Handle 1D case
            1 => Ok(self.fwd.process(data.as_slice_mut().unwrap())),

            // Handle 2D Case
            2 => Ok({
                // Iterate through rows
                for mut row in data.rows_mut() {
                    self.fwd.process(row.as_slice_mut().expect("invalid row"));
                }

                // Transpose xy -> yx
                let mut scratch: ArrayD<Complex<T>> = ArrayD::<Complex<T>>::zeros(IxDyn(&[S]));
                transpose_inplace(
                    data.as_slice_mut().unwrap(),
                    scratch.as_slice_mut().unwrap(),
                    S,
                    S,
                );

                // Iterate through columns (now rows)
                for mut col in data.rows_mut() {
                    self.fwd.process(col.as_slice_mut().expect("invalid col"));
                }

                // Tranpose yx -> xy
                transpose_inplace(
                    data.as_slice_mut().unwrap(),
                    scratch.as_slice_mut().unwrap(),
                    S,
                    S,
                );
            }),

            // Handle 3D case
            3 => Ok({
                // Iterate through z-axis
                for mut zlane in data.lanes_mut(Axis(2)) {
                    self.fwd.process(zlane.as_slice_mut().expect("invalid z"));
                }

                // Transpose xyz -> zxy
                let mut scratch: ArrayD<Complex<T>> = ArrayD::<Complex<T>>::zeros(IxDyn(&[S, S]));
                transpose_inplace(
                    data.as_slice_mut().unwrap(),
                    scratch.as_slice_mut().unwrap(),
                    S,
                    S * S,
                );

                // Iterate through y-axis
                for mut ylane in data.lanes_mut(Axis(2)) {
                    self.fwd.process(ylane.as_slice_mut().expect("invalid y"));
                }

                // Transpose zxy -> yzx
                transpose_inplace(
                    data.as_slice_mut().unwrap(),
                    scratch.as_slice_mut().unwrap(),
                    S,
                    S * S,
                );

                // Iterate through x-axis
                for mut xlane in data.lanes_mut(Axis(2)) {
                    self.fwd.process(xlane.as_slice_mut().expect("invalid x"));
                }

                // Transpose yzx -> xyz
                transpose_inplace(
                    data.as_slice_mut().unwrap(),
                    scratch.as_slice_mut().unwrap(),
                    S,
                    S * S,
                );
            }),

            // Not implemented for D != 1, 2, 3
            k => Err(MSMError::IncorrectNumDumensions(k)),
        }
    }

    /// This function handles the inverse FFT operation for Dimension = 1, 2, 3
    pub fn inverse(&self, data: &mut Array<Complex<T>, Dim<[usize; K]>>) -> Result<(), MSMError>
    where
        T: FftNum + ValueFrom<usize> + ScalarOperand + Float + std::ops::AddAssign,
        Complex<T>: std::ops::DivAssign + std::ops::AddAssign + ScalarOperand,
        Dim<[usize; K]>: Dimension + IntoDimension,
    {
        // Ensure data provided is of the size supported
        assert_eq!(data.shape(), &[S; K]);

        // Dimension dependent forward
        match K {

            // Handle 1D case
            1 => Ok(self.inv.process(data.as_slice_mut().unwrap())),

            // Handle 2D case
            2 => Ok({

                // Iterate through rows
                for mut row in data.rows_mut() {
                    self.inv.process(row.as_slice_mut().expect("invalid row"));
                }

                // Transpose xy -> yx
                let mut scratch: ArrayD<Complex<T>> = ArrayD::<Complex<T>>::zeros(IxDyn(&[S]));
                transpose_inplace(
                    data.as_slice_mut().unwrap(),
                    scratch.as_slice_mut().unwrap(),
                    S,
                    S,
                );

                // Iterate through col (now rows)
                for mut col in data.rows_mut() {
                    self.inv.process(col.as_slice_mut().expect("invalid col"));
                }

                // Transpose yx -> xy
                transpose_inplace(
                    data.as_slice_mut().unwrap(),
                    scratch.as_slice_mut().unwrap(),
                    S,
                    S,
                );
            }),

            // Handle 3D Case
            3 => Ok({

                // Iterate through z-axis
                for mut zlane in data.lanes_mut(Axis(2)) {
                    self.inv.process(zlane.as_slice_mut().expect("invalid z"));
                }

                // Transpose xyz -> zxy
                let mut scratch: ArrayD<Complex<T>> = ArrayD::<Complex<T>>::zeros(IxDyn(&[S, S]));
                transpose_inplace(
                    data.as_slice_mut().unwrap(),
                    scratch.as_slice_mut().unwrap(),
                    S,
                    S * S,
                );

                // Iterate through y-axis
                for mut ylane in data.lanes_mut(Axis(2)) {
                    self.inv.process(ylane.as_slice_mut().expect("invalid y"));
                }

                // Transpose zxy --> yzx
                transpose_inplace(
                    data.as_slice_mut().unwrap(),
                    scratch.as_slice_mut().unwrap(),
                    S,
                    S * S,
                );

                // Iterate through x-axis
                for mut xlane in data.lanes_mut(Axis(2)) {
                    self.inv.process(xlane.as_slice_mut().expect("invalid x"));
                }

                // Transpose yzx -> xyz
                transpose_inplace(
                    data.as_slice_mut().unwrap(),
                    scratch.as_slice_mut().unwrap(),
                    S,
                    S * S,
                );
            }),
            k => Err(MSMError::IncorrectNumDumensions(k)),
        }
    }
}

pub enum MSMError {
    IncorrectNumDumensions(usize),
}

#[test]
fn create_fft_object() {
    const FFT_SIZE: usize = 16;
    const DIM: usize = 1;

    // 32 bit floating point
    type T = f32;
    let _fft_f32 = FftObject::<T, DIM, FFT_SIZE>::new();

    // 64 bit floating point
    type U = f64;
    let _fft_f64 = FftObject::<U, DIM, FFT_SIZE>::new();
}

#[test]
fn test_fft_object_1_d_usage() {
    // Set FFT parameters
    const FFT_SIZE: usize = 16;
    const DIM: usize = 1;
    type T = f32;

    // Create FFT Object
    let fft = FftObject::<T, DIM, FFT_SIZE>::new();

    // Define data to operate on
    let mut data = arr1(&[Complex::<T> { re: 1.0, im: 0.0 }; FFT_SIZE]);
    let orig = data.clone();

    // Carry out fwd + inv FFT
    fft.fwd.process(data.as_slice_mut().expect("invalid data"));
    fft.inv.process(data.as_slice_mut().expect("invalid data"));

    // Renormalize
    data = data / FFT_SIZE as T;

    // Check that sum of norm of elementwise difference is tiny or zero
    assert_abs_diff_eq!(
        data.map(|x| x.norm()).sum(),
        orig.map(|x| x.norm()).sum(),
        epsilon = 1e-9
    );
}

#[test]
fn test_fft_object_2_d_usage() {
    // Set FFT parameters
    const FFT_SIZE: usize = 2;
    const DIM: usize = 2;
    type T = f32;

    // Create FFT Object
    let fft = FftObject::<T, DIM, FFT_SIZE>::new();

    // Define data to operate on
    let mut data = arr2(&[[Complex::<T> { re: 1.0, im: 0.0 }; FFT_SIZE]; FFT_SIZE]);
    let orig = data.clone();

    // Carry out fwd + inv FFT
    fft.forward(&mut data);
    fft.inverse(&mut data);

    // Check that sum of norm of elementwise difference is tiny or zero
    assert_abs_diff_eq!(
        data.map(|x| x.norm()).sum(),
        orig.map(|x| x.norm()).sum(),
        epsilon = 1e-9
    );
}

#[test]
fn test_fft_object_3_d_usage() {
    // Set FFT parameters
    const FFT_SIZE: usize = 16;
    const DIM: usize = 3;
    type T = f32;

    // Create FFT Object
    let fft = FftObject::<T, DIM, FFT_SIZE>::new();

    // Define data to operate on
    // Much easier to test on non-uniform data w/ no symmetries
    let mut data = arr3(&[[[Complex::<T> { re: 1.0, im: 0.0 }; FFT_SIZE]; FFT_SIZE]; FFT_SIZE]);
    data[[0, 0, 0]] = Complex::<T> { re: 1.0, im: 0.0 };
    data[[0, 0, 1]] = Complex::<T> { re: 2.0, im: 0.0 };
    data[[0, 1, 0]] = Complex::<T> { re: 3.0, im: 0.0 };
    data[[0, 1, 1]] = Complex::<T> { re: 4.0, im: 0.0 };
    data[[1, 0, 0]] = Complex::<T> { re: 5.0, im: 0.0 };
    data[[1, 0, 1]] = Complex::<T> { re: 6.0, im: 0.0 };
    data[[1, 1, 0]] = Complex::<T> { re: 7.0, im: 0.0 };
    data[[1, 1, 1]] = Complex::<T> { re: 8.0, im: 0.0 };
    let orig = data.clone();

    // Carry out fwd + inv FFT
    fft.forward(&mut data);
    fft.inverse(&mut data);

    // Check that sum of norm of elementwise difference is tiny or zero
    assert_abs_diff_eq!(
        data.map(|x| x.norm()).sum(),
        orig.map(|x| x.norm()).sum(),
        epsilon = 1e-9
    );
}
