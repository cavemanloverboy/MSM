use ndarray::{array, arr1, arr2, arr3, s, Array, Dim, Dimension, IntoDimension, ShapeBuilder, Data, Axis, ScalarOperand};
use rustfft::{num_complex::Complex, FftPlanner, FftNum};
use transpose::transpose;
use conv::prelude::*;

/// This struct is intended to be initialized at the
/// beginning of a simulation. It holds the forward
/// and inverse fft objects that perform FFTs.
pub struct FftObject<T, const K: usize>
where 
    T: FftNum + ValueFrom<usize> + ScalarOperand,
{
    // Planner
    pub planner: rustfft::FftPlanner<T>,
    // Forward operation
    pub fwd: std::sync::Arc<dyn rustfft::Fft<T>>,
    // Inverse operation
    pub inv: std::sync::Arc<dyn rustfft::Fft<T>>,
    // Keep size on record
    pub size: usize,
}

impl <T, const K: usize> FftObject<T, K> 
where
    T: FftNum + ValueFrom<usize> + ScalarOperand,
    Complex<T>: std::ops::DivAssign,
{
    /// This `FftObject` constructor takes in a `size` (which is stored)
    /// and returns a struct containing the fft planner (`rustfft::FftPlanner<T>`)
    /// along with the associated forward and inverse operators.
    pub fn new(size: usize) -> Self {

        // Create planner of type T
        let mut planner = FftPlanner::<T>::new();
        // Create forward and inverse plans
        let fwd = planner.plan_fft_forward(size);
        let inv = planner.plan_fft_inverse(size);

        // Pack into struct and return
        FftObject {
            planner,
            fwd,
            inv,
            size,
        }
    }
    
    /// This function takes
    pub fn forward(&self, data: &mut Array<Complex<T>, Dim<[usize; K]>>) -> Result<(), MSMError>
    where
        T: FftNum + ValueFrom<usize> + ScalarOperand,
        Complex<T>: std::ops::DivAssign<T>,
        Dim<[usize; K]> : Dimension + IntoDimension,
    
    {

        // Ensure data provided is of the size supported
        assert_eq!(data.shape(), &[self.size; K]);
        
        // Forward needs division
        //*data = *data/self.size.value_as::<T>().unwrap();
        //data = &mut data.map(|x| x/self.size.value_as::<T>().unwrap());
        *data /= &array![Complex{ re: self.size.pow(K as u32).value_as::<T>().unwrap(), im: T::zero()} ];
        //*data /= array![ Complex{ re: self.size.value_as::<T>().unwrap(), im: T::zer;

        // Dimension dependent forward
        match K {
            1 => Ok(self.fwd.process(data.as_slice_mut().unwrap())),
            2 => Ok({
                    // Iterate through rows
                    for mut row in data.rows_mut() {
                        self.fwd.process(row.as_slice_mut().expect("invalid row"));
                    }

                    // Transpose rows and columns
                    transpose::transpose(data.clone().as_slice().unwrap(), data.as_slice_mut().unwrap(), self.size, self.size);
            
                    // Iterate through rows (prev columns)
                    for mut row in data.rows_mut() { 
                        self.fwd.process(row.as_slice_mut().expect("invalid row post-transpose"));
                    }
                }),
            3 => Ok({
                    // Iterate through axis 0
                    for mut xlane in data.lanes_mut(Axis(0)) {
                        self.fwd.process(xlane.as_slice_mut().expect("invalid x-axis"));
                    }
                    // Iterate through axis 1
                    for mut ylane in data.lanes_mut(Axis(1)) { 
                        self.fwd.process(ylane.as_slice_mut().expect("invalid y-axis"));
                    }
                    // Iterate through axis 2
                    for mut zlane in data.lanes_mut(Axis(2)) { 
                        self.fwd.process(zlane.as_slice_mut().expect("invalid z-axis"));
                    }
                }),
            k => Err(MSMError::IncorrectNumDumensions(k))
        }
    }

    /// This function takes
    pub fn inverse(&self, data: &mut Array<Complex<T>, Dim<[usize; K]>>) -> Result<(), MSMError>
    where
        T: FftNum + ValueFrom<usize> + ScalarOperand,
        Complex<T>: std::ops::DivAssign<T>,
        Dim<[usize; K]> : Dimension + IntoDimension,
    
    {

        // Ensure data provided is of the size supported
        assert_eq!(data.shape(), &[self.size; K]);
        
        // Forward needs division
        //*data = *data/self.size.value_as::<T>().unwrap();
        //data = &mut data.map(|x| x/self.size.value_as::<T>().unwrap());
        //*data /= &array![Complex{ re: self.size.value_as::<T>().unwrap(), im: T::zero()} ];
        //*data /= array![ Complex{ re: self.size.value_as::<T>().unwrap(), im: T::zer;

        // Dimension dependent forward
        match K {
            1 => Ok(self.inv.process(data.as_slice_mut().unwrap())),
            2 => Ok({
                    // Iterate through rows
                    for mut row in data.rows_mut() {
                        self.inv.process(row.as_slice_mut().expect("invalid row"));
                    }

                    // Transpose rows and columns
                    transpose::transpose(data.clone().as_slice().unwrap(), data.as_slice_mut().unwrap(), self.size, self.size);
            
                    // Iterate through rows (prev columns)
                    for mut row in data.rows_mut() { 
                        self.inv.process(row.as_slice_mut().expect("invalid row post-transpose"));
                    }
                }),
            3 => Ok({
                    // Iterate through axis 0
                    for mut xlane in data.lanes_mut(Axis(0)) {
                        self.inv.process(xlane.as_slice_mut().expect("invalid x-axis"));
                    }
                    // Iterate through axis 1
                    for mut ylane in data.lanes_mut(Axis(1)) { 
                        self.inv.process(ylane.as_slice_mut().expect("invalid y-axis"));
                    }
                    // Iterate through axis 2
                    for mut zlane in data.lanes_mut(Axis(2)) { 
                        self.inv.process(zlane.as_slice_mut().expect("invalid z-axis"));
                    }
                }),
            k => Err(MSMError::IncorrectNumDumensions(k))
        }
        

    }
}

//fn fourier_transform()


pub enum MSMError {
    IncorrectNumDumensions(usize)
}


#[test]
fn create_fft_object() {

    const FFT_SIZE: usize = 16;
    const DIM: usize = 1;

    // 32 bit floating point
    type T = f32;
    let fft_f32 = FftObject::<T, DIM>::new(FFT_SIZE);

    // 64 bit floating point
    type U = f64;
    let fft_f64 = FftObject::<U, DIM>::new(FFT_SIZE);

}

#[test]
fn test_fft_object_1_d_usage() {

    // Set FFT parameters
    const FFT_SIZE: usize = 16;
    const DIM: usize = 1;
    type T = f32;

    // Create FFT Object
    let fft = FftObject::<T, DIM>::new(FFT_SIZE);
    
    // Define data to operate on
    let mut data = arr1(&[Complex::<T> {re: 1.0, im: 0.0}; FFT_SIZE]);
    let orig = data.clone();
    
    // Carry out fwd + inv FFT
    fft.fwd.process(data.as_slice_mut().expect("invalid data"));
    fft.inv.process(data.as_slice_mut().expect("invalid data"));

    // Renormalize
    data = data/FFT_SIZE as T;

    // Check that sum of norm of elementwise difference is tiny or zero
    assert!((data-orig).map(|x| x.norm()).sum() < 1e-9);
}

#[test]
fn test_fft_object_2_d_usage() {

    // Set FFT parameters
    const FFT_SIZE: usize = 16;
    const DIM: usize = 2;
    type T = f32;

    // Create FFT Object
    let fft = FftObject::<T, DIM>::new(FFT_SIZE);
    
    // Define data to operate on
    let mut data = arr2(&[[Complex::<T> {re: 1.0, im: 0.0}; FFT_SIZE]; FFT_SIZE]);
    let orig = data.clone();
    
    // Carry out fwd + inv FFT
    fft.forward(&mut data);
    fft.inverse(&mut data);

    // Check that sum of norm of elementwise difference is tiny or zero
    assert!((data-orig).map(|x| x.norm()).sum() < 1e-9);
}


