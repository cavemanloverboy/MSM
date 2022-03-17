use num::{Complex, Float};
use arrayfire::{Array, HasAfEnum, FloatingPoint, Dim4};


pub fn complex_constant<T>(
    value: Complex<T>,
    shape: (u64, u64, u64, u64),
) -> Array<Complex<T>>
where 
    T: Float + FloatingPoint,
    Complex<T>: HasAfEnum + FloatingPoint
{
    let array_values = vec![value; (shape.0*shape.1*shape.2*shape.3) as usize];
    let dims = Dim4::new(&[shape.0, shape.1, shape.2, shape.3]);
    Array::new(&array_values, dims)
}