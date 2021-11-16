use pnn::nn::Network;

fn main() {
    let mut net = Network::from_darknet(String::from("./cfgs/tests/yolov4-csp.cfg")).unwrap();
    net.set_batchsize(4).unwrap();
    net.render(String::from("./render/test.dot")).unwrap();
}