#[test]
fn test_io() {
    use arrayfire::Array;
    use msm::utils::io::*;
    use std::io::Read;

    // Specifiy size, type
    const S: usize = 4;
    const K: usize = 3;
    type T = f32;

    // Build arrayfire array
    let values = [1.0 as T; S.pow(K as u32)];
    let shape = [S as u64, S as u64, S as u64, 1];
    let array = Array::new(&values, array_to_dim4(shape));

    // Host array
    let mut host = vec![1.0 as T; array.elements()];
    array.host(&mut host);

    // Build nd_array for npy serialization
    let host: ndarray::Array1<T> = ndarray::ArrayBase::from_vec(host);
    let host = host.into_shape(array_to_tuple(shape)).unwrap();

    // Write to npz
    use ndarray_npy::NpzWriter;
    use std::fs::File;
    let mut npz = NpzWriter::new(File::create("data.npz").unwrap());
    npz.add_array("host", &host);
    npz.finish();

    // Write to npy
    npy::to_file("data.npy", host).unwrap();

    // Read
    use npy::NpyData;
    let mut buf = vec![];
    std::fs::File::open("data.npy")
        .unwrap()
        .read_to_end(&mut buf)
        .unwrap();
    let data: NpyData<T> = NpyData::from_bytes(&buf).unwrap();
}

#[test]
fn test_npz() {
    use ndarray::{array, Array1, Array2};
    use ndarray_npy::NpzWriter;
    use std::fs::File;

    let mut npz = NpzWriter::new(File::create("arrays.npz").unwrap());
    let a: Array2<i32> = array![[1, 2, 3], [4, 5, 6]];
    let b: Array1<i32> = array![7, 8, 9];
    npz.add_array("a", &a);
    npz.add_array("b", &b);
    npz.finish();
}
