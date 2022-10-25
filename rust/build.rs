use std::error::Error;
// use std::io::copy;
use std::env;
// use std::fs::File;
// use reqwest;
// use futures::executor::block_on;

/// URL to 3.8.0 linux installer
#[cfg(target_os = "linux")]
const LINUX_3_8_0_INSTALLER: &'static str =
    "https://arrayfire.s3.amazonaws.com/3.8.0/ArrayFire-v3.8.0_Linux_x86_64.sh";

/// URL to 3.7.2 mac installer
#[cfg(target_os = "macos")]
const MAC_3_8_0_INSTALLER: &'static str =
    "https://arrayfire.s3.amazonaws.com/3.7.2/ArrayFire-3.7.2_OSX_x86_64.pkg";

fn main() -> Result<(), Box<dyn Error>> {
    println!(
        "cargo:rustc-env=AF_PATH={}",
        env::current_dir().unwrap().join("arrayfire").display()
    );
    env::set_var("AF_PATH", env::current_dir().unwrap().join("arrayfire"));
    println!(
        "cargo:rustc-env=LD_LIBRARY_PATH={}:{}/lib64",
        env::var("LD_LIBRARY_PATH").unwrap(),
        &env::var("AF_PATH").unwrap()
    );
    println!(
        "cargo:warning= these are the vars: {}, {}",
        env::var("AF_PATH").unwrap(),
        env!("LD_LIBRARY_PATH")
    );

    // Build arrayfire if necessary
    if std::path::Path::exists("./arrayfire".as_ref()) {
        // User seems to have arrayfire built
    } else {
        build_arrayfire()?;
    }

    // Set env vars

    Ok(())
}

#[cfg(target_os = "macos")]
fn build_arrayfire() {
    todo!(format!("write mac installer, {MAC_3_8_0_INSTALLER}"))
}

#[cfg(target_os = "linux")]
fn build_arrayfire() -> Result<(), Box<dyn Error>> {
    // Download installer if necessary
    if !std::path::Path::exists("af_installer.sh".as_ref()) {
        download_file(LINUX_3_8_0_INSTALLER)?;
    }

    // Run installer
    println!("installer");
    println!(
        "{:?}",
        std::process::Command::new("bash")
            .args(["af_installer.sh", "--include-subdir", "--skip-license"])
            .output()
            .expect("failed to build arrayfire")
    );

    // delete installer
    std::fs::remove_file("af_installer.sh").expect("failed to delete installer file");

    Ok(())
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
fn build_arrayfire() {
    panic!(
        "Building Arrayfire on your operating system is not supported by the MSM crate. \
            If on Windows, see the Arrayfire docs for building 3.8.0 on Windows."
    )
}

#[tokio::main(flavor = "current_thread")]
async fn download_file(url: &'static str) -> Result<(), Box<dyn Error>> {
    std::process::Command::new("wget")
        .args([url, "--output-document=af_installer.sh"])
        .output()
        .expect("failed");

    Ok(())
}
