use std::error::Error;
use std::io::copy;
use std::env;
use std::fs::File;
use reqwest;
use futures::executor::block_on;


/// URL to 3.8.0 linux installer
#[cfg(target_os = "linux")]
const LINUX_3_8_0_INSTALLER: &'static str = "https://arrayfire.s3.amazonaws.com/3.8.0/ArrayFire-v3.8.0_Linux_x86_64.sh";

/// URL to 3.7.2 mac installer
#[cfg(target_os = "macos")]
const MAC_3_8_0_INSTALLER: &'static str = "https://arrayfire.s3.amazonaws.com/3.7.2/ArrayFire-3.7.2_OSX_x86_64.pkg";


fn main() -> Result<(), Box<dyn Error>> {

    // Build arrayfire if necessary
    if std::path::Path::exists("./arrayfire".as_ref()) {
        // User seems to have built arrayfire
    } else {
        build_arrayfire()?;
    }

    // Set env vars
    #[cfg(any(target_os = "macos", target_os="linux"))]
    env::set_var("AF_PATH", env::current_dir().unwrap().join("arrayfire"));
    env::set_var("LD_LIBRARY_PATH", env::var("LD_LIBRARY_PATH").unwrap() + ":$AF_PATH");

    Ok(())
}


#[cfg(target_os = "macos")]
fn build_arrayfire() {
    todo!(format!("write mac installer, {MAC_3_8_0_INSTALLER}"))
}

#[cfg(target_os = "linux")]
fn build_arrayfire() -> Result<(), Box<dyn Error>> {
    
    // Download installer
    download_file(LINUX_3_8_0_INSTALLER)?;

    // Run installer
    std::process::Command::new("bash ./ArrayFire-v3.8.0_Linux_x86_64.sh")
            .args(["--include-subdir", "--prefix=./", "--skip-license"])
            .output()
            .expect("failed to build arrayfire");

    Ok(())
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
fn build_arrayfire() {
    panic!("Building Arrayfire on your operating system is not supported by the MSM crate. \
            If on Windows, see the Arrayfire docs for building 3.8.0 on Windows.")
}



#[tokio::main(flavor = "current_thread")]
async fn download_file(url: &'static str) -> Result<(), Box<dyn Error>> {

    // Get response from url
    let response = reqwest::get(url).await?;

    // File save destination
    let mut dest = File::create("./af_installer.sh")?;

    // Copy content to file
    let content = response.text().await?;
    copy(&mut content.as_bytes(), &mut dest)?;

    Ok(())
}