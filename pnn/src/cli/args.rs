use clap::{Parser, Subcommand};
use crate::nn::{
    BuildError
};
use crate::cudnn::{
    cudnnDataType
};

#[derive(Parser, Debug)]
#[clap(about, version, author)]
struct Args {

    #[clap(subcommand)]
    command: Commands,
}


#[derive(Subcommand, Debug)]
enum Commands {
    /// Do performance benchmark
    Benchmark {
        /// Load as TensorRT engine.
        #[clap(long)]
        trt: bool,

        /// Path to weights 
        #[clap(short, long)]
        weights: String,
        
        /// Path to config
        #[clap(short, long)]
        config: String,
        
        /// Batchsize. Default value 1
        #[clap(short, long, default_value_t = 1)]
        batchsize: usize,

        /// Input file
        #[clap(short, long)]
        input: String,

        /// Output render file
        #[clap(short, long)]
        output: Option<String>,

        /// Render window during work
        #[clap(short, long)]
        show: bool,

        /// Build HALF precision engine
        #[clap(short, long)]
        half: bool,

        /// Confidence threshold
        #[clap(short, long, default_value_t = 0.45)]
        threshold: f32,

        /// Confidence threshold
        #[clap(short, long, default_value_t = 0.45)]
        iou_tresh: f32,

        /// Confidence threshold
        #[clap(short, long, default_value_t = String::from("./cfgs/tests/coco.names"))]
        classes_file: String
    },
    /// Build TensorRT engine file
    Build {
        /// Path to weights 
        #[clap(short, long)]
        weights: String,
        
        /// Path to config
        #[clap(short, long)]
        config: String,
        
        /// Batchsize. Default value 1
        #[clap(short, long, default_value_t = 1)]
        batchsize: usize,
        
        /// Output engine
        #[clap(short, long)]
        output: Option<String>,

        /// Build HALF precision engine
        #[clap(short, long)]
        half: bool
    }
}

pub fn build_engine(weights: String, config: String, batchsize: usize, output: Option<String>, half: bool) -> Result<(), crate::nn::BuildError> {
    assert!(batchsize > 0, "Batchsize should be greater 0");

    let mut net = crate::nn::Network::from_darknet(&config)?;
    let (data_type, suffix) = if half {(cudnnDataType::HALF, "fp16")} else {(cudnnDataType::FLOAT, "fp32")};
    let engine = output.or_else(||{Some(format!("yolo_bs{}_{}.engine", batchsize, suffix))});

    net.build_trt(batchsize, data_type, &weights, engine)?;

    Ok(())
}

pub fn parse() {
    let args = Args::parse();
    match args.command {
        Commands::Build{weights, config, batchsize, output, half} => build_engine(weights, config, batchsize, output, half).unwrap(),
        Commands::Benchmark{trt, weights, config, batchsize, output, half, input, show, threshold, iou_tresh, classes_file} => crate::cli::demo(
            input, 
            config, 
            weights, 
            classes_file, 
            half,
            batchsize, 
            threshold,
            iou_tresh,
            show,
            output, 
            trt).unwrap()
    }
}