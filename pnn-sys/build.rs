use bindgen::Builder;
use cc::Build;

fn main() {
    println!("cargo:rustc-link-lib=cuda");
    println!("cargo:rustc-link-lib=cudnn");
    println!("cargo:rustc-link-lib=cudart");

    let cuda_dir = option_env!("CUDA_PATH").unwrap_or("/usr/local/cuda");
    println!("cargo:rustc-link-search={}/lib64", cuda_dir);
    let bindings = Builder::default()
        .header("./cudnn/cudnn_api.h")
        .clang_args(&[format!("-I{}/include", cuda_dir)])
        .size_t_is_usize(true)
        .generate()
        .expect("Unable to generate bindings");

    bindings.write_to_file("./src/bindings.rs")
        .expect("Couldn't write bindings!");

    Build::new()
        .include("/usr/local/include/opencv4")
        .cuda(true)
        .flag("-cudart=shared")
        .flag("--use_fast_math")
        .flag("-arch=sm_80") // TODO: Add autodiscovery
        .files(&["./cuda/mish.cu", "./cuda/upsample.cu", "./cuda/utils.cpp"])
        .compile("kernels.a");

    let opencv_libs = ["imgcodecs", "core", "imgproc"];
    for lib in opencv_libs {
        println!("cargo:rustc-link-lib=opencv_{}", lib);
    }
}