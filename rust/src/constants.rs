#![allow(dead_code)]

/// Poisson constant
pub const POIS_CONST: f64 = 4.0 * std::f64::consts::PI * 4.49e-12;

//pub const POIS_CONST: f64 = 4.0 * std::f64::consts::PI;

// Reduced Planck constant
pub const HBAR: f64 = 1.757e-90_f64; // This is in solar masses * kpc^2 / Myr

//pub const HBAR: f64 = 1.757e-90_f64 * (1.115 * 1e66 * 1e22); // This is in solar masses * kpc^2 / Myr, scaled by new particle mass for 10^10 particles
//pub const HBAR: f64 = 1.757e-90_f64 * (1.115 * 1e67 * 1e22); // This is in solar masses * kpc^2 / Myr, scaled by new particle mass for 10^12 particles
//pub const HBAR: f64 = 1.0;
