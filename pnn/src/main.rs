use pnn::parsers::parse_file;
use pnn::cudnn::{Tensor, cudaError, cudaMalloc, cudaFree, cudnnCreate, cudnnDestroy};
use pnn::nn::{LayerShape, Shape};

fn main() {
    println!("Hello world!");
    // let kek = cudnnCreate().unwrap();
    // std::thread::sleep(std::time::Duration::from_secs(5));
    // cudnnDestroy(kek);
    // std::thread::sleep(std::time::Duration::from_secs(5));
    {
        let shape = Box::new(LayerShape::from_nchw(1, 3, 1920, 1080));
        let tensor = Tensor::new(shape);
        std::thread::sleep(std::time::Duration::from_secs(5));
    }
    

}
