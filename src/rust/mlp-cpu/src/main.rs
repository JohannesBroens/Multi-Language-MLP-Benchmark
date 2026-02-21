fn main() {
    let args = mlp_common::parse_args();
    let dataset = mlp_common::load_dataset(&args);
    println!(
        "Dataset: {}  ({} samples, {} features, {} classes)",
        args.dataset, dataset.num_samples, dataset.input_size, dataset.output_size
    );
    println!("mlp-cpu: MLP training not yet implemented.");
}
