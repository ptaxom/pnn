use pnn::nn::Network;

fn main() {
    let net = Network::from_darknet(String::from("./cfgs/tests/yolov4-csp.cfg")).unwrap();
    net.render(String::from("./render/test.dot")).unwrap();
}