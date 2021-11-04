use cmake::Config;
use std::env;

fn main() {
    if env::var("CARGO_FEATURE_STATIC").is_ok() {
        if cfg!(target_os = "windows") {
            #[cfg(target_os = "windows")]
            let ceres = vcpkg::find_package("ceres").unwrap();
            #[cfg(target_os = "windows")]
            let glog = vcpkg::find_package("glog").unwrap();
            #[cfg(target_os = "windows")]
            let gflags = vcpkg::find_package("gflags").unwrap();
            #[cfg(target_os = "windows")]
            let mut b = cxx_build::bridge("src/solve_silent.rs");
            #[cfg(target_os = "windows")]
            b.flag_if_supported("/std:c++14")
                .includes(
                    ceres
                        .include_paths
                        .iter()
                        .chain(glog.include_paths.iter())
                        .chain(gflags.include_paths.iter())
                        .map(|include| include.to_str().unwrap()),
                )
                .include("src")
                .compile("autocxx-ceres");
            #[cfg(target_os = "windows")]
            println!("cargo:rerun-if-changed=src/lib.rs");
            #[cfg(target_os = "windows")]
            println!("cargo:rerun-if-changed=ceres-solver");

            #[cfg(target_os = "windows")]
            println!("cargo:rustc-link-lib=shlwapi")
        } else {
            let ceres = Config::new("ceres-solver")
                .define("EXPORT_BUILD_DIR", "ON")
                .define("CXX_THREADS", "ON")
                .define("BUILD_TESTING", "OFF")
                .define("BUILD_BENCHMARKS", "OFF")
                .define("MINIGLOG", "ON")
                .define("LAPACK", "OFF")
                .define("CUSTOM_BLAS", "OFF")
                .define("SCHUR_SPECIALIZATIONS", "OFF")
                .define("BUILD_EXAMPLES", "OFF")
                .define("LIB_SUFFIX", "")
                .define("SUITESPARSE", "OFF")
                .define("CXSPARSE", "OFF")
                .build();
            println!("cargo:rustc-link-search=native={}/lib", ceres.display());
            let profile = std::env::var("PROFILE").unwrap();
            let lib_name = match profile.as_str() {
                "debug" => "ceres-debug",
                "release" => "ceres",
                _ => "ceres",
            };
            println!("cargo:rustc-link-lib=static={}", lib_name);
            let sysinclude3 = std::path::PathBuf::from("/usr/include/eigen3");
            let sysinclude = std::path::PathBuf::from("/usr/include/eigen3");
            let localinclude3 = std::path::PathBuf::from("/usr/local/include/eigen3");
            let localinclude = std::path::PathBuf::from("/usr/local/include/eigen");
            let targetminiglog = std::path::PathBuf::from(format!(
                "{}/include/ceres/internal/miniglog",
                ceres.display()
            ));
            let targetinclude = std::path::PathBuf::from(format!("{}/include", ceres.display()));
            let mut b = cxx_build::bridge("src/solve_silent.rs");
            b.flag_if_supported("-std=c++14")
                .flag_if_supported("-Wno-unused-parameter")
                .include(sysinclude3)
                .include(sysinclude)
                .include(localinclude3)
                .include(localinclude)
                .include(targetminiglog)
                .include(targetinclude)
                .include("src")
                .compile("autocxx-ceres");
            println!("cargo:rerun-if-changed=src/lib.rs");
            println!("cargo:rerun-if-changed=ceres-solver");
        }
    } else {
        println!("cargo:rustc-link-lib=ceres");
        let sysinclude3 = std::path::PathBuf::from("/usr/include/eigen3");
        let sysinclude = std::path::PathBuf::from("/usr/include/eigen3");
        let localinclude3 = std::path::PathBuf::from("/usr/local/include/eigen3");
        let localinclude = std::path::PathBuf::from("/usr/local/include/eigen");
        let mut b = cxx_build::bridge("src/solve_silent.rs");
        b.flag_if_supported("-std=c++14")
            .flag_if_supported("-Wno-unused-parameter")
            .include(sysinclude3)
            .include(sysinclude)
            .include(localinclude3)
            .include(localinclude)
            .include("src")
            .compile("autocxx-ceres");

        println!("cargo:rerun-if-changed=src/lib.rs");
        println!("cargo:rerun-if-changed=ceres-solver");
    }
    let target = env::var("TARGET").unwrap();
    if target.contains("apple") {
        println!("cargo:rustc-link-lib=dylib=c++");
    } else if target.contains("linux") {
        println!("cargo:rustc-link-lib=dylib=stdc++");
    } else if target.contains("win") {
    } else {
        unimplemented!()
    }
}
