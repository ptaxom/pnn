use pnn::nn::Network;

fn main() {
    let net = Network::from_darknet(String::from("./cfgs/tests/yolov4-csp.cfg")).unwrap();
    println!("{}", net.layers.len());
}