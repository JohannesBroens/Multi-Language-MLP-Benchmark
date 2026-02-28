fn main() {
    let cuda_path =
        std::env::var("CUDA_PATH").unwrap_or_else(|_| "/usr/local/cuda".to_string());

    println!("cargo:rustc-link-search=native={}/lib64", cuda_path);
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=cublas");

    // Compile CUDA kernels
    cc::Build::new()
        .cuda(true)
        .flag("-O3")
        .flag("-arch=native")
        .file("src/kernels.cu")
        .compile("cnn_cublas_kernels");

    println!("cargo:rustc-link-lib=dylib=cudart");
}
