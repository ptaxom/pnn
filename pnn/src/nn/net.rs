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
}

pub struct Network {
    // List of layers
    layers: Vec<LayerRef>,
    // Layer connectivity graph
    adjency_matrix: Vec<Vec<Link>>,
    // Input layers 
    input_layers: Vec<LayerRef>,
    // Output layers
    output_layers: Vec<LayerRef>
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

    // Link layer with index origin to layer with index target
    fn link_layers(&mut self, origin: usize, target: usize) {
        self.adjency_matrix[origin][target] = Link::Forward;
        self.adjency_matrix[target][origin] = Link::Backward;
    }

    fn reset_adjency_matrix(&mut self) {
        let mut adjency_matrix: Vec<Vec<Link>> = Vec::new();
        let n_layers = self.layers.len();
        for _ in 0..n_layers {
            let mut row: Vec<Link> = Vec::new();
            row.resize(n_layers, Link::None);
            adjency_matrix.push(row);
        }
        self.adjency_matrix = adjency_matrix;
    }

    // Default implementation to link generic layer
    fn link_layer(&mut self, index: usize) -> Result<(), DeserializationError> {
        let layer = &self.layers[index];
        let input_indices = layer.as_ref().borrow().input_indices(index)?;
        for src_id in input_indices {
            self.link_layers(src_id, index);
        }
        self.link_layers(index, index + 1);
        Ok(())
    }

    fn build_adjency_matrix(&mut self) -> Result<(), DeserializationError> {
        self.reset_adjency_matrix();

        let n_layers = self.layers.len();
        let mut removed_routes: Vec<usize> = Vec::new();
        
        for layer_pos in 0..n_layers-1 {
           let layer = &self.layers[layer_pos];
           let layer_type = layer.as_ref().borrow().layer_type();
           match layer_type {
            LayerType::Input => self.link_layers(layer_pos, layer_pos + 1),
            LayerType::Route => { // Separatly handle RouteLayer, because of it can be just redirection or mixin
                let input_indices = layer.as_ref().borrow().input_indices(layer_pos)?;
                if input_indices.len() == 1 { // If it just redirection from some layer to another
                    removed_routes.insert(0, layer_pos);
                    self.link_layers(input_indices[0], layer_pos + 1);
                } else {
                    self.link_layer(layer_pos)?;
                }
            }
            _ => {
                self.link_layer(layer_pos)?;
            }
           }
        }

        // Dropping rows/cols with optimized routes
        for &index in &removed_routes {
            self.adjency_matrix.remove(index);
            self.layers.remove(index);
        }
        for row in &mut self.adjency_matrix {
            for &index in &removed_routes {
                row.remove(index);
            }
        }

        Ok(())
    }

    pub fn from_darknet(darknet_cfg: String) -> Result<Self, Box<dyn std::error::Error>> {
        let config = parse_file(&darknet_cfg)?;
        if config.len() < 2 {
            return Err(
                Box::new(DeserializationError{description: String::from("Network must contain at least input and output layers")})
            )
        }
        let mut layers: Vec<LayerRef> = Vec::new();
        let mut input_layers: Vec<LayerRef> = Vec::new();
        let mut output_layers: Vec<LayerRef> = Vec::new();
        for layer_cfg in config {
            let layer = Network::parse_layer(layer_cfg)?;
            let layer = Rc::new(RefCell::new(layer));

            layers.push(layer.clone());
            match layer.as_ref().borrow().layer_type() {
                LayerType::Input => input_layers.push(layer.clone()),
                LayerType::YoloLayer => output_layers.push(layer.clone()),
                _ => ()
            };
        }

        let adjency_matrix: Vec<Vec<Link>> = Vec::new();
        let n_layers = layers.len();
        for _ in 0..n_layers {
            let mut row: Vec<Link> = Vec::new();
            row.resize(n_layers, Link::None);
        }
        let mut net = Network{layers, adjency_matrix, input_layers, output_layers};
        net.build_adjency_matrix()?;
        Ok(net)
    }

    pub fn render(&self, dot_path: String) -> Result<(), std::io::Error> {
        use tabbycat::attributes::*;
        use tabbycat::{AttrList, GraphBuilder, GraphType, Identity, StmtList, Edge, SubGraph};
        use std::fs::write;

        let names: Vec<String> = self.layers.iter().map(|x| {
            x.as_ref().borrow().name()
        }).collect();
        let n_layers = self.layers.len();

        let mut statements = StmtList::new();
        for i in 0..n_layers {
            statements = statements.add_node(
                Identity::id(&names[i]).unwrap(),
                None,
                Some(AttrList::new().add_pair(shape(Shape::Box)))
            );
        }

        for head_id in 0..n_layers-1 {
            for target_id in head_id+1..n_layers {
                match self.adjency_matrix[head_id][target_id] {
                    Link::Forward => {
                        statements = statements.add_edge(
                            Edge::head_node(Identity::id(&names[head_id]).unwrap(), None)
                                .arrow_to_node(Identity::id(&names[target_id]).unwrap(), None)
                        )
                    },
                    _ => ()
                }
            }
        }

        let graph = GraphBuilder::default()
            .graph_type(GraphType::DiGraph)
            .strict(false)
            .id(Identity::id("G").unwrap())
            .stmts(statements)
            .build()
            .unwrap();
        write(dot_path, graph.to_string())?;
        Ok(())
    }

}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn load_yolov4_csp() {
        let net = Network::from_darknet(String::from("../cfgs/tests/yolov4-csp.cfg")).unwrap();
        assert_eq!(net.layers.len(), 161);
    }
}