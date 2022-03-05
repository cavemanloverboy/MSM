use ndarray::parallel::prelude::*;
use ndarray::{arr3, Array, Axis};

#[test]
#[ignore]
fn simple_parallel() {
    // // create some 3d array
    let mut my_3_d_array = arr3(&[[[0, 1], [2, 3]], [[4, 5], [6, 7]]]);
}

#[test]
fn verbatim_test() {
    let a = Array::linspace(0., 63., 64).into_shape((4, 16)).unwrap();
    let mut sums = Vec::new();
    a.axis_iter(Axis(0))
        .into_par_iter()
        .map(|row| row.sum())
        .collect_into_vec(&mut sums);

    assert_eq!(sums, [120., 376., 632., 888.]);
}
