fn main() {
    let cuda_path =
        std::env::var("CUDA_PATH").unwrap_or_else(|_| "/usr/local/cuda".to_string());

    println!("cargo:rustc-link-search=native={}/lib64", cuda_path);
    println!("cargo:rustc-link-lib=dylib=cudart");
    // NO cuBLAS — we use custom kernels only

    cc::Build::new()
        .cuda(true)
        .flag("-O3")
        .flag("-arch=native")
        .file("src/kernels.cu")
        .compile("mlp_raw_kernels");

    println!("cargo:rustc-link-lib=dylib=cudart");
}
