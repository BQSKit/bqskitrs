fn main() -> Result<(), std::io::Error> {
    // Homebrew/macOS gcc don't add libgfortran to the rpath,
    // so we manually go prodding around for it here
    if cfg!(target_os = "windows") {
        #[cfg(target_os = "windows")]
        let _lapack = vcpkg::Config::new()
            .target_triplet("x64-windows-static-md")
            .find_package("clapack")
            .unwrap();
        println!("cargo:rustc-link-lib=static=lapack");
    }
    Ok(())
}