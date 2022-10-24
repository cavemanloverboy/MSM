#[test]
fn test_variance_of_arrayfire_randn() {
    use arrayfire::{add, imag, mul, random_normal, real, var_all, Array, Dim4, RandomEngine};
    use num::Complex;

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
        false,
    );

    println!("{:?}", var_all(&real(&normals), false));
    println!("{:?}", var_all(&imag(&normals), false));
}

#[test]
fn test_seeds() {
    use arrayfire::{add, imag, mul, random_normal, real, var_all, Array, Dim4, RandomEngine};
    use num::Complex;

    const S: usize = 8;
    const K: u32 = 1;
    let dims = Dim4::new(&[S.pow(K) as u64, 1, 1, 1]);

    let seed_1 = Some(0);
    let engine_1 = RandomEngine::new(arrayfire::RandomEngineType::PHILOX_4X32_10, seed_1);
    let seed_2 = Some(1);
    let engine_2 = RandomEngine::new(arrayfire::RandomEngineType::PHILOX_4X32_10, seed_2);
    let normals_1: Array<f64> = random_normal(dims, &engine_1);
    let normals_2: Array<f64> = random_normal(dims, &engine_2);
    let mut host_1 = vec![0.0; S.pow(K)];
    let mut host_2 = vec![0.0; S.pow(K)];
    normals_1.host(&mut host_1);
    normals_2.host(&mut host_2);

    println!("{:?}", host_1);
    println!("{:?}", host_2);
}
