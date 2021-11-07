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
        let shape = Box::new(LayerShape::from_nchw(1, 1, 1, 4));
        let mut tensor = Tensor::new(shape).unwrap();
        let data: Vec<f32> = vec![1., 2., 3., 4.];
        tensor.load(&data).unwrap();
        println!("{}", tensor);
        std::thread::sleep(std::time::Duration::from_secs(5));
    }
    

}
