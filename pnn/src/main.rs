use human_panic::setup_panic;
use pnn::cli::parse;

fn main() {
    setup_panic!();
    parse();

}

// pnn::cli::demo(
//     &String::from("../models/yolo_test.mp4"),
//     &String::from("./cfgs/tests/yolov4-csp.cfg"),
//     &String::from("../models/yolov4-csp.weights"),
//     &String::from("./cfgs/tests/coco.names"),
//     &cudnnDataType::HALF,
//     4,
//     0.3,
//     0.3
// ).unwrap();