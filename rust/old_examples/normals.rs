use arrayfire::{random_normal, RandomEngine, Array, mul, add, var_all, real, imag, Dim4};
use num::Complex;

fn main() {

    const S: u64 = 128;
    const K: u32 = 3;

    let seed = Some(0);
    let engine = RandomEngine::new(arrayfire::RandomEngineType::PHILOX_4X32_10, seed);

    let dims = Dim4::new(&[S.pow(K), 1, 1, 1]);
    let normals: Array<Complex<f64>> = add(
        
        &mul::<Array<f64>, Complex<f64>>(
            &random_normal(dims, &engine),
            &Complex::new(1.0, 0.0),
            true,
        ),
        &mul::<Array<f64>, Complex<f64>>(
            &random_normal(dims, &engine),
            &Complex::new(0.0, 1.0),
            true,
        ),
        false
    );


    println!("{:?}", var_all(&real(&normals), false));
    println!("{:?}", var_all(&imag(&normals), false));
}