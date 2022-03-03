use ndarray::{array, arr1, arr2, arr3, s, Array, ShapeBuilder};
use rustfft::{num_complex::Complex, FftPlanner, FftNum};
use transpose::transpose;


/// This struct is intended to be initialized at the
/// beginning of a simulation. It holds the forward
/// and inverse fft objects that process FFTs.
pub struct FftObject<T, const K: usize>
where T: FftNum
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
where T: FftNum
{
    pub fn new(size: usize) -> Self {

        let mut planner = FftPlanner::<T>::new();
        let fwd = planner.plan_fft_forward(size);
        let inv = planner.plan_fft_inverse(size);

        FftObject {
            planner,
            fwd,
            inv,
            size,
        }
    }
    
    pub fn forward(self, data: &mut [Complex<T>]) {


        assert_eq!(data.len(), self.size);

        self.fwd.process(data)


    }
}

//fn fourier_transform()



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
    fft.fwd.process(data.as_slice_mut().expect("invalid data"));
    fft.inv.process(data.as_slice_mut().expect("invalid data"));

    // Renormalize
    data = data/FFT_SIZE as T;

    // Check that sum of norm of elementwise difference is tiny or zero
    assert!((data-orig).map(|x| x.norm()).sum() < 1e-9);
}