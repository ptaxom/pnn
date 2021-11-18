use pnn::nn::Network;
use pnn::cudnn::cudnnDataType;
use std::time::{Duration, Instant};
use std::thread::sleep;

fn main() {
    let mut net = Network::from_darknet(String::from("./cfgs/tests/yolov4-csp.cfg")).unwrap();
    let bs = 1;
    net.set_batchsize(bs).unwrap();
    net.load_darknet_weights(&String::from("../models/yolov4-csp.weights")).unwrap();
    net.build(cudnnDataType::FLOAT).unwrap();
    println!("Builded yolo");
    
    net.load_image(String::from("../models/test2.jpg"), 0).unwrap();
    net.forward_debug().unwrap();

    let N = 1;
    let mut t: f32 = 0.;
    for iter in 0..N {
        let now = Instant::now();
        net.forward().unwrap();
        t += now.elapsed().as_secs_f32();
    }
    let fps = N as f32 / t * bs as f32;
    println!("Estimated FPS = {}[{}]", fps, t);
    std::thread::sleep_ms(5000);
    // net.render(String::from("./render/test.dot")).unwrap();
}