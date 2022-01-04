use human_panic::setup_panic;
use pnn::cli::parse;

fn main() {
    setup_panic!();
    parse();
}