use pnn::nn::Network;
use pnn::cudnn::cudnnDataType;
use std::time::{Instant};

fn old_main() {
    let impath = String::from("../models/test2.jpg");
    let classes = pnn::parsers::load_classes("./cfgs/tests/coco.names").unwrap();
    let mut net = Network::from_darknet(&String::from("./cfgs/tests/yolov4-csp.cfg")).unwrap();
    let bs = 1;
    net.build_cudnn(bs, cudnnDataType::FLOAT, Some(String::from("../models/yolov4-csp.weights"))).unwrap();
    println!("Builded yolo");
    
    net.load_image(impath.clone(), 0).unwrap();
    // net.load_bin(&String::from("./debug/darknet/input_0.bin")).unwrap();
    // net.forward_debug().unwrap();

    let n = 1;
    let mut t: f32 = 0.;
    for _ in 0..n {
        let now = Instant::now();
        net.forward().unwrap();
        t += now.elapsed().as_secs_f32();
    }
    let fps = n as f32 / t * bs as f32;
    println!("Estimated FPS = {}[{}]", fps, t);

    net.set_detections_ops(0.25, 0.1);
    let preds = net.get_detections().unwrap();
    // for b_id in 0..bs {
    //     for bbox in &preds[b_id] {
    //         println!("{}", &bbox);
    //     }
    // }
    pnn::cudnn::render_bboxes(&impath, &preds[0], &classes, &String::from("Result")).unwrap();
    net.render(String::from("./render/test.dot")).unwrap();
}

fn main() {
    // #TODO: Check multibatch sync
    pnn::cli::demo(
        &String::from("../models/yolo_test.mp4"),
        &String::from("./cfgs/tests/yolov4-csp.cfg"),
        &String::from("../models/yolov4-csp.weights"),
        &String::from("./cfgs/tests/coco.names"),
        &cudnnDataType::HALF,
        4,
        0.3,
        0.3
    ).unwrap();
}