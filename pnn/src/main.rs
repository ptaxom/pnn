use pnn::parsers::parse_file;
use pnn::cudnn::{Tensor, cudaError, cudaMalloc, cudaFree, cudnnCreate, cudnnDestroy, add};
use pnn::nn::{LayerShape, Shape};

fn main() {
    // println!("Hello world!");
    // let kek = cudnnCreate().unwrap();
    // std::thread::sleep(std::time::Duration::from_secs(5));
    // cudnnDestroy(kek);
    // std::thread::sleep(std::time::Duration::from_secs(5));
    {
        let context = cudnnCreate().unwrap();
        let shape = Box::new(LayerShape::from_nchw(1, 1, 1, 4));

        let mut A = Tensor::new(shape.clone()).unwrap();
        let data: Vec<f32> = vec![1., 2., 3., 4.];
        A.load(&data).unwrap();

        let mut B = Tensor::new(shape).unwrap();
        let data: Vec<f32> = vec![4., 3., 2., 1.];
        B.load(&data).unwrap();

        println!("A  : {}", A);
        println!("B  : {}", B);

        add(context, &A, &mut B);

        println!("A+B: {}", B);
        std::thread::sleep(std::time::Duration::from_secs(5));
    }
    

}
