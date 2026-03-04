fn main() {
    let cuda_path =
        std::env::var("CUDA_PATH").unwrap_or_else(|_| "/usr/local/cuda".to_string());

    println!("cargo:rustc-link-search=native={}/lib64", cuda_path);
    println!("cargo:rustc-link-lib=dylib=cudart");
    // NO cuBLAS — we use custom tiled matmul kernels only

    // cuDNN (pip-installed, v9)
    let home = std::env::var("HOME").unwrap();
    let cudnn_root = format!(
        "{}/.local/lib/python3.12/site-packages/nvidia/cudnn",
        home
    );
    println!("cargo:rustc-link-search=native={}/lib", cudnn_root);
    println!("cargo:rustc-link-lib=dylib=cudnn");
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}/lib", cudnn_root);

    println!("cargo:rerun-if-changed=src/kernels.cu");

    cc::Build::new()
        .cuda(true)
        .flag("-O3")
        .flag("-arch=native")
        .include(format!("{}/include", cudnn_root))
        .file("src/kernels.cu")
        .compile("cnn_raw_kernels");

    println!("cargo:rustc-link-lib=dylib=cudart");
}
