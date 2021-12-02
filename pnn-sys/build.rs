use bindgen::Builder;
use cc::Build;

fn main() {
    println!("cargo:rustc-link-lib=cuda");
    println!("cargo:rustc-link-lib=cudnn");
    println!("cargo:rustc-link-lib=cudart");

    println!("cargo:rustc-link-lib=nvinfer");
    println!("cargo:rustc-link-lib=nvinfer_plugin");

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
        .cpp(true)
        .flag("-cudart=shared")
        .flag("--use_fast_math")
        .flag("-arch=sm_80") // TODO: Add autodiscovery
        .files(&["./cuda/mish.cu", "./cuda/upsample.cu", 
                 "./cuda/utils.cpp", "./cuda/convert.cu",
                 "./cuda/blas.cu", "./cuda/demo.cpp",
                 "./trt/mish.cu",
                 ])
        .compile("kernels.a");

    Build::new()
        .include(format!("{}/include", cuda_dir))
        .cpp(true)
        .files(&["./trt/activation.hpp", "./trt/activation.cpp",
                "./trt/builder.hpp", "./trt/builder.cpp",
                "./trt/engine.hpp", "./trt/engine.cpp",
                "./trt/utils.hpp", "./trt/utils.cpp",
                "./trt/trt.h", "./trt/trt.cpp",
                "./trt/mish.h",
        ])
        .compile("trt.a");

    let opencv_libs = ["imgcodecs", "core", "imgproc", "highgui", "videoio"];
    for lib in opencv_libs {
        println!("cargo:rustc-link-lib=opencv_{}", lib);
    }
}