pub mod complex;
pub mod error;
pub mod fft;
pub mod grid;
pub mod io;

#[cfg(feature = "expanding")]
type IndependentVariable = f64;

/// This function does a single step of the rk4 algorithm in 1D.
/// It is tested indirectly through the tests for solving the Friedmann Equation
/// for the evolution of the scale factor.
#[cfg(feature = "expanding")]
pub(crate) fn rk4<F>(
    // Function that evaluates derivative
    f: F,
    // Current time
    tn: IndependentVariable,
    // Current function value
    yn: f64,
    // Timestep
    h: f64,
    // Option to provide already-computed derivative at tn (i.e. f(tn,yn))
    derivative: Option<f64>,
) -> f64
where
    F: Fn(IndependentVariable, f64) -> f64,
{
    // Evaluate the derivative function at (tn, yn)
    let k1: f64 = derivative.unwrap_or_else(|| f(tn, yn));

    // Evaluate the derivative funciton at (tn + h/2, y + h*k1/2)
    let k2: f64 = f(tn + h / 2.0, yn + h * k1 / 2.0);

    // Evaluate the derivative funciton at (tn + h/2, y + h*k2/2)
    let k3: f64 = f(tn + h / 2.0, yn + h * k2 / 2.0);

    // Evaluate the derivative funciton at (tn + h, y + h*k3)
    let k4: f64 = f(tn + h, yn + h * k3);

    // Return new value as per RK4
    yn + h * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
}
