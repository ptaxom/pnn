use pnn::nn::Network;
use pnn::cudnn::cudnnDataType;

fn main() {
    let mut net = Network::from_darknet(String::from("./cfgs/tests/yolov4-csp.cfg")).unwrap();
    net.set_batchsize(4).unwrap();
    net.build(cudnnDataType::FLOAT).unwrap();
    std::thread::sleep_ms(5000);
    // net.render(String::from("./render/test.dot")).unwrap();
}