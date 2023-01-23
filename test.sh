printf "\n----- FEATURES: NONE -----\n\n"
cargo run --bin msm-simulator --release --no-default-features -- --toml examples/spherical-tophat.toml 
cargo run --bin msm-synthesizer --release --no-default-features -- --toml examples/spherical-tophat.toml 

printf "\n---- FEATURES: EXPANDING -----\n\n"
cargo run --bin msm-simulator --release --no-default-features --features=expanding -- --toml examples/spherical-tophat-cosmo.toml
cargo run --bin msm-synthesizer --release --no-default-features --features=expanding -- --toml examples/spherical-tophat-cosmo.toml 

printf "\n----- FEATURES: REMOTE-STORAGE -----\n\n"
cargo run --bin msm-simulator --release --no-default-features --features=remote-storage -- --toml examples/spherical-tophat-cosmo.toml

printf "\n----- FEATURES: EXPANDING + REMOTE-STORAGE -----\n\n"
cargo run --bin msm-simulator --release --no-default-features --features=expanding,remote-storage -- --toml examples/spherical-tophat-cosmo.toml
