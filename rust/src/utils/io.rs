use arrayfire::Dim4;




/// This is a helper function to turn a length 4 array (required by Dim4) into a tuple,
/// which is required by ndarray::Array's .reshape() method
pub fn array_to_tuple(
    dim4: [u64; 4],
) -> (usize, usize, usize, usize) {
    (dim4[0] as usize, dim4[1] as usize, dim4[2] as usize, dim4[3] as usize)
}

/// This is a helper function to turn a length 4 array (required by Dim4) into a Dim4,
pub fn array_to_dim4(
    dim4: [u64; 4],
) -> Dim4 {
    Dim4::new(&dim4)
}