use pnn::parsers::parse_file;


fn main() {
    let filename = String::from("./cfgs/tests/base.cfg");
    let config = parse_file(&filename).unwrap();
    
    println!("Hello wor");
}
