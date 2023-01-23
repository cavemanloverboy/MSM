/// Poisson constant
pub const POIS_CONST: f64 = 4.0 * std::f64::consts::PI * 4.49e-12;

// Reduced Planck constant
pub const HBAR: f64 = 1.757e-90_f64; // This is in solar masses * kpc^2 / Myr

/// Multiplicative factor used to turn h to H in units of 1/Myr.
/// (i.e. 100 km/s/Mpc converted to 1/Myr)
pub const LITTLE_H_TO_BIG_H: f64 = 1.022e-4;
