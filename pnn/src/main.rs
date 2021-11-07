use pnn::parsers::parse_file;
use pnn::cudnn::{Tensor, cudaError, cudaMalloc, cudaFree, cudnnCreate, cudnnDestroy};

fn main() {
    println!("Hello world!");
    let kek = cudnnCreate().unwrap();
    std::thread::sleep(std::time::Duration::from_secs(5));
    cudnnDestroy(kek);
    std::thread::sleep(std::time::Duration::from_secs(5));
}
