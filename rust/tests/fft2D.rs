use ndarray::{arr2, Array};
use rustfft::{num_complex::Complex, FftPlanner};
use transpose::transpose;

#[test]
fn test_2_d_fft_with_ndarray() {
    // Perform a forward FFT of size FFT_SIZE (per dimension)
    const FFT_SIZE: usize = 4;
    type T = f64;

    // Plan forward and inverse ffts
    let mut planner = FftPlanner::<T>::new();
    let fft = planner.plan_fft_forward(FFT_SIZE);
    let ifft = planner.plan_fft_inverse(FFT_SIZE);

    // Define data, save copy for comparison
    // This is a FFT_SIZE x FFT_SIZE array
    let mut my_array: [[Complex<T>; FFT_SIZE]; FFT_SIZE] =
        [[Complex::<T> { re: 1.0, im: -1.0 }; FFT_SIZE]; FFT_SIZE];
    my_array[0][2] = Complex::<T> { re: 5.0, im: -5.0 };
    let mut buffer: Array<Complex<T>, _> = arr2(&my_array); //array![my_array];
    let orig = buffer.clone();

    println!("{:?}", buffer);

    // Forward FFT
    {
        for mut row in buffer.rows_mut() {
            println!("{}", &row);
            fft.process(row.as_slice_mut().expect("invalid row"));
        }
        println!("{}", "-".repeat(40));
        transpose(
            buffer.clone().as_slice().unwrap(),
            buffer.as_slice_mut().unwrap(),
            FFT_SIZE,
            FFT_SIZE,
        );

        for mut row in buffer.rows_mut() {
            println!("{}", &row);
            fft.process(row.as_slice_mut().expect("invalid row post-transpose"));
        }
    }
    println!("{}", "-".repeat(40));
    // Inverse FFT
    {
        for mut row in buffer.rows_mut() {
            println!("{}", &row);
            ifft.process(row.as_slice_mut().expect("invalid row"));
        }

        println!("{}", "-".repeat(40));
        transpose(
            buffer.clone().as_slice().unwrap(),
            buffer.as_slice_mut().unwrap(),
            FFT_SIZE,
            FFT_SIZE,
        );

        for mut row in buffer.rows_mut() {
            println!("{}", &row);
            //println!("{:?}", row.as_slice_mut());
            ifft.process(row.as_slice_mut().expect("invalid row post-transpose"));
        }
    }

    println!("{}", "-".repeat(40));
    for row in buffer.rows() {
        println!("{}", &row);
    }
    buffer = buffer / (FFT_SIZE as T) / (FFT_SIZE as T); //* arr2(&[[Complex{ re: 1.0/(FFT_SIZE as T)/(FFT_SIZE as T), im: 1.0/(FFT_SIZE as T)/(FFT_SIZE as T) }; FFT_SIZE]; FFT_SIZE]);//::Array<Complex<T>,_>();
    println!("{}", "-".repeat(40));
    for row in buffer.rows() {
        println!("{}", &row);
    }
    println!("{:?}", buffer.map(|x| x.norm()));

    assert!((buffer - orig).map(|x| x.norm()).sum() < 1e-6)
}
