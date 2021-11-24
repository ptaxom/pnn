use pnn::nn::Network;
use pnn::cudnn::cudnnDataType;
use std::time::{Instant};

fn main() {
    let impath = String::from("../models/test2.jpg");
    let classes = pnn::parsers::load_classes("./cfgs/tests/coco.names").unwrap();
    let mut net = Network::from_darknet(String::from("./cfgs/tests/yolov4-csp.cfg")).unwrap();
    let bs = 4;
    net.set_batchsize(bs).unwrap();
    net.load_darknet_weights(&String::from("../models/yolov4-csp.weights")).unwrap();
    net.build(cudnnDataType::HALF).unwrap();
    println!("Builded yolo");
    
    net.load_image(impath.clone(), 0).unwrap();
    // net.load_bin(&String::from("./debug/darknet/input_0.bin")).unwrap();
    // net.forward_debug().unwrap();

    let n = 1000;
    let mut t: f32 = 0.;
    for _ in 0..n {
        let now = Instant::now();
        net.forward().unwrap();
        t += now.elapsed().as_secs_f32();
    }
    let fps = n as f32 / t * bs as f32;
    println!("Estimated FPS = {}[{}]", fps, t);

    
    let preds = net.get_yolo_predictions(0.25, 0.1).unwrap();
    // for b_id in 0..bs {
    //     for bbox in &preds[b_id] {
    //         println!("{}", &bbox);
    //     }
    // }
    pnn::cudnn::render_bboxes(&impath, &preds[0], &classes, &String::from("Result")).unwrap();
    net.render(String::from("./render/test.dot")).unwrap();
}