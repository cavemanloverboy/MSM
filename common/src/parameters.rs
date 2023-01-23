use regex::Regex;
use serde::{Deserialize, Serialize};

use crate::{
    constants::*,
    error::CommonError,
    ics::{InitialConditions, SamplingScheme},
};

#[derive(Serialize, Deserialize)]
pub struct TomlParameters {
    /// Physical length of box
    pub axis_length: f64,
    /// Start (and current) time of simulation
    pub time: Option<f64>,
    /// End time of simulation
    pub final_sim_time: f64,
    /// Safety factor
    pub cfl: f64,
    /// Number of data dumps
    pub num_data_dumps: u32,
    /// Total mass in the system
    pub total_mass: f64,
    /// Particle mass (use this or hbar_ = HBAR / particle_mass)
    pub particle_mass: Option<f64>,
    /// Specify number of particles. If this is used, `particle_mass` is ignored and HBAR const is ignored.
    pub ntot: Option<f64>,
    /// HBAR / particle mass (use this or particle_mass)
    pub hbar_: Option<f64>,
    /// Name of simulation (used for directories)
    pub sim_name: String,
    /// Where to begin checking for cutoff (this is a parameter in [0,1])
    pub k2_cutoff: f64,
    /// How much of mass to use as threshold for constitutes aliasing (P(k > k2_cutoff * k2))
    pub alias_threshold: f64,
    /// Dimensionality of grid
    pub dims: usize,
    /// Number of grid cells per dim
    pub size: usize,
    /// Initial Conditions
    pub ics: InitialConditions,
    /// Sampling Parameters
    pub sampling: Option<TomlSamplingParameters>,

    /// Flag indicating whether we output potential
    #[serde(default = "bool::default")]
    pub output_potential: bool,

    /// Cosmological Parameters
    #[cfg(feature = "expanding")]
    pub cosmology: CosmologyParameters,

    #[cfg(feature = "remote-storage")]
    pub remote_storage_parameters: RemoteStorageParameters,
}

#[cfg(feature = "remote-storage")]
#[derive(Serialize, Deserialize, Clone)]
pub struct RemoteStorageParameters {
    /// Keypair to use for remote storage. NOTE: supply a path, and the deserializer will attempt to deserialize
    #[serde(alias = "keypair_path")]
    pub keypair: String,

    /// Storage account to use
    pub storage_account: String,
}

#[derive(Serialize, Deserialize, Debug, Clone, Copy)]
/// A helper struct. Deserialized from a simulation's toml file.
#[cfg(feature = "expanding")]
pub struct CosmologyParameters {
    /// Omega_matter now in units of critical density
    pub omega_matter_now: f64,

    /// Omega_radiation now in units of critical density
    pub omega_radiation_now: f64,

    /// Hubble constant now (little h)
    pub h: f64,

    /// Initial redshift at which simulation is initialized
    pub z0: f64,

    /// Solver parameter (maximum dloga change in one timestep). Default value of [`DEFAULT_MAX_DLOGA`].
    pub max_dloga: Option<f64>,
}

#[derive(Serialize, Deserialize, Debug, PartialEq)]
pub struct TomlSamplingParameters {
    pub scheme: SamplingScheme,
    #[serde(deserialize_with = "deserialize_seeds")]
    pub seeds: Vec<u64>,
}

/// This function reads toml files
pub fn read_toml(path: &str) -> Result<TomlParameters, CommonError> {
    // Read toml config file
    let toml_contents: &str =
        &std::fs::read_to_string(path).map_err(|_| CommonError::TomlReadError {
            path: path.to_string(),
        })?;

    // Return parsed toml from str
    toml::from_str(toml_contents).map_err(|e| CommonError::TomlParseError {
        msg: format!("{e:?}"),
    })
}

fn deserialize_seeds<'de, D>(deserializer: D) -> Result<Vec<u64>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    // Get the string
    let Ok(parsed_string) = <&str>::deserialize(deserializer) else {
        panic!("fialed to parsed");
    };

    parse_seeds(parsed_string).map_err(serde::de::Error::custom)
}

#[test]
fn test_regex_range_exclusive() {
    let sample = "0..=55";
    let seeds = parse_seeds(sample);
    assert_eq!(seeds, Ok((0..=55).collect()));
}

#[test]
fn test_regex_to() {
    let sample = "0 to 55";
    let seeds = parse_seeds(sample);
    assert_eq!(seeds, Ok((0..=55).collect()));
}

#[test]
fn test_regex_comma_separated() {
    let sample = "[1, 3]";
    let seeds = parse_seeds(sample);
    assert_eq!(seeds, Ok([1, 3].into_iter().collect()));

    let sample = "1, 3";
    let seeds = parse_seeds(sample);
    assert_eq!(seeds, Ok([1, 3].into_iter().collect()));
}

/// NOTE: this compiles the regex internally,
/// so if we ever use this multiple times it will be slow/inefficient.
fn parse_seeds(s: &str) -> Result<Vec<u64>, &'static str> {
    let r1 = Regex::new(r"\d+..=\d+").unwrap();
    let r2 = Regex::new(r"\d+ to \d+").unwrap();
    let r3 = Regex::new(r"(\d+)[^,]?+").unwrap();
    match s {
        // Range Inclusive (a..=b)
        _ if r1.is_match(s) => {
            // Get start and end points
            let [start, end]: [u64; 2] = s
                .split("..=")
                .map(str::parse::<u64>)
                .map(Result::unwrap)
                .collect::<Vec<u64>>()
                .try_into()
                .unwrap();

            // Collect seeds
            Ok((start..=end).collect())
        }

        // Custom syntax (a to b)
        _ if r2.is_match(s) => {
            // Get start and end points
            let [start, end]: [u64; 2] = s
                .split(" to ")
                .map(str::parse::<u64>)
                .map(Result::unwrap)
                .collect::<Vec<u64>>()
                .try_into()
                .unwrap();

            // Collect seeds
            Ok((start..=end).collect())
        }

        // Comma separated digits
        _ if r3.is_match(s) => {
            // Collect and parse collection
            Ok(r3
                .find_iter(s)
                .map(|_match| {
                    println!("{}", _match.as_str().trim_matches(']'));
                    str::parse::<u64>(_match.as_str().trim_matches(']'))
                })
                .map(Result::unwrap)
                .collect::<Vec<u64>>())
        }

        _ => {
            return Err(
                "seeds did not match expected patterns: low..=high, low to high, [s1, s2, s3]",
            )
        }
    }
}

#[cfg(feature = "expanding")]
pub fn get_supercomoving_boxsize(
    hbar_: f64,
    cosmo_params: CosmologyParameters,
    axis_length: f64,
) -> f64 {
    use crate::constants::*;

    let initial_proper_boxsize = axis_length;
    let initial_scale_factor = (1.0 + cosmo_params.z0).recip();
    let comoving_boxsize = initial_proper_boxsize / initial_scale_factor;
    // (3/2 * H^2 * Omo)^(1/4) * (1/hbar)^(1/2)
    ((1.5 * cosmo_params.omega_matter_now * (LITTLE_H_TO_BIG_H * cosmo_params.h).powi(2)).sqrt()
        / hbar_)
        .sqrt()
        * comoving_boxsize
}

pub fn determine_pmass_hbar_(toml: &TomlParameters) -> (f64, f64) {
    let particle_mass;
    let hbar_;
    if let Some(ntot) = toml.ntot {
        // User has specified the total mass and ntot.
        // So, the particle mass can be derived.

        particle_mass = toml.total_mass / ntot;
        hbar_ = toml.hbar_.unwrap_or_else(|| {
            println!("hbar_ not specified, using HBAR / particle_mass.");
            HBAR / particle_mass
        });
    } else if let Some(p_mass) = toml.particle_mass {
        // User has specified the total mass and particle mass.
        // So, the ntot can be derived, as can hbar_ if not specified.

        particle_mass = p_mass;
        hbar_ = toml.hbar_.unwrap_or_else(|| {
            println!("hbar_ not specified, using HBAR / particle_mass.");
            HBAR / particle_mass
        });
    } else if let Some(hbar_tilde) = toml.hbar_ {
        // User has specified the total mass and hbar_.
        // So, the ntot and particle_mass can be derived.

        hbar_ = hbar_tilde;
        particle_mass = HBAR / hbar_
        // ntot isn't actually stored but is determined via total_mass / particle_mass;
    } else {
        panic!(
            "You must specify the total mass and either exactly one of ntot (total number \
              of particles) or particle_mass, or hbar_tilde ( hbar / particle_mass ). Note: you
              can specify hbar_tilde in addition to one of the first two if you'd like to change
              the value of planck's constant itself."
        )
    }
    (particle_mass, hbar_)
}
