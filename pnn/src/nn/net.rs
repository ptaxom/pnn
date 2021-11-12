use super::layers::*;
use std::{
    rc::Rc,
    cell::RefCell,
    collections::HashMap
};
use crate::parsers::*;

pub type LayerRef = Rc<RefCell<Box<dyn Layer>>>;

#[derive(Debug, Clone)]
pub enum Link {
    Forward,
    Backward,
    None,
    Optimized
}

pub struct Network {
    // List of layers
    pub layers: Vec<LayerRef>,
    // Layer connectivity graph
    adjency_matrix: Vec<Vec<Link>>,
}


impl Network {

    fn parse_layer(config: HashMap<String, String>) -> Result<Box<dyn Layer>, DeserializationError>  {
        let key = config.get("type").unwrap();
        let layer_type = LayerType::from(key);
        match  layer_type {
            LayerType::Input => InputLayer::from_config(config),
            LayerType::Convolutional => ConvolutionalLayer::from_config(config),
            LayerType::Shortcut => ShortcutLayer::from_config(config),
            LayerType::Upsample => UpsampleLayer::from_config(config),
            LayerType::Route => RouteLayer::from_config(config),
            LayerType::YoloLayer => YoloLayer::from_config(config),
            LayerType::Maxpool => MaxpoolLayer::from_config(config),            
            LayerType::Unknown => return Err(
                DeserializationError{description: format!("Couldnt deserialize config for '{}'", key)}
            ),
        }
    }

    pub fn from_darknet(darknet_cfg: String) -> Result<Self, Box<dyn std::error::Error>> {
        let config = parse_file(&darknet_cfg)?;
        let mut layers: Vec<LayerRef> = Vec::new();
        for layer_cfg in config {
            let layer = Network::parse_layer(layer_cfg)?;
            layers.push(Rc::new(RefCell::new(layer)));
        }

        let adjency_matrix: Vec<Vec<Link>> = Vec::new();
        let n_layers = layers.len();
        for _ in 0..n_layers {
            let mut row: Vec<Link> = Vec::new();
            row.resize(n_layers, Link::None);
        }

        Ok(Network{layers, adjency_matrix})
    }

}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn load_yolov4_csp() {
        let net = Network::from_darknet(String::from("../cfgs/tests/yolov4-csp.cfg")).unwrap();
        assert_eq!(net.layers.len(), 176);
    }
}