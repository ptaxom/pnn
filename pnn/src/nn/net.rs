use super::layers::*;
use std::{
    rc::Rc,
    cell::RefCell,
    collections::HashMap
};
use crate::parsers::*;
use crate::nn::errors::*;
use crate::cudnn::{cudnnHandle_t, cudnnCreate, cudnnDataType, cudaDeviceSynchronize};
use crate::nn::ops::*;
use crate::nn::{Engine, CUDNNEngine};

pub type LayerRef = Rc<RefCell<Box<dyn Layer>>>;

#[derive(Debug, Clone, PartialEq)]
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
    // Batchsize
    batchsize: Option<usize>,
    // Input size(height, width)
    size: (usize, usize),
    // prob_thresh, nms_threshold
    detection_ops: Option<(f32, f32)>,
    // Inference(or train) engine
    engine: Option<Box<dyn Engine>>
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
            LayerType::Yolo => YoloLayer::from_config(config),
            LayerType::Maxpool => MaxpoolLayer::from_config(config),            
            LayerType::Unknown => return Err(
                DeserializationError(format!("Couldnt deserialize config for '{}'", key))
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
            LayerType::Yolo => (),
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

    fn check_inited(&self) -> Result<(), RuntimeError> {
        if self.batchsize == None {
            return Err(RuntimeError::Other(String::from("Batchsize is not setted")))
        }
        if self.context == None {
            return Err(RuntimeError::Other(String::from("Network is not builded")))
        }
        Ok(())
    }

    pub fn from_darknet(darknet_cfg: &String) -> Result<Self, BuildError> {
        let config = parse_file(darknet_cfg)?;
        if config.len() < 2 {
            return Err(
                BuildError::Deserialization(DeserializationError(String::from("Network must contain at least input and output layers")))
            )
        }
        let mut size = (0, 0);
        let mut yolo_heads = Vec::new();

        let mut layers: Vec<LayerRef> = Vec::new();
        let mut n_inputs = 0;
        for layer_cfg in config {
            if layer_cfg.get("type").unwrap() == "net" {
                size = (
                    parse_numerical_field::<usize>(&layer_cfg, "height", true, None).map_err(|e| {BuildError::Deserialization(e)})?.unwrap(),
                    parse_numerical_field::<usize>(&layer_cfg, "width", true, None).map_err(|e| {BuildError::Deserialization(e)})?.unwrap(),
                );
            }
            // net should be first in config file, otherwise this code fail
            if size == (0, 0) {
                return Err(BuildError::Deserialization(DeserializationError(String::from("Fields height and widht are mandatory"))));
            }


            let layer = Network::parse_layer(layer_cfg).map_err(|e| {
                BuildError::Deserialization(e)
            })?;
            let layer = Rc::new(RefCell::new(layer));
            if layer.as_ref().borrow().layer_type() == LayerType::Input {
                n_inputs += 1;
            }
            if layer.as_ref().borrow().layer_type() == LayerType::Yolo {
                yolo_heads.push(layer.clone());
            }

            layers.push(layer.clone());
        }

        if n_inputs != 1 {
            return Err(
                BuildError::Deserialization(DeserializationError(String::from("Supported only exact one input layer")))
            )
        }

        let adjency_matrix: Vec<Vec<Link>> = Vec::new();
        let n_layers = layers.len();
        for _ in 0..n_layers {
            let mut row: Vec<Link> = Vec::new();
            row.resize(n_layers, Link::None);
        }
        let batchsize = None;
        let detection_ops = None;
        let engine = None;
        let mut net = Network{layers, adjency_matrix, batchsize, size, detection_ops, engine};
        net.build_adjency_matrix().map_err(|e| {
            BuildError::Deserialization(e)
        })?;
        Ok(net)
    }

    pub fn get_detection_ops(&self) -> Option<(f32, f32)> {
        return self.detection_ops.clone()
    }

    pub fn set_detections_ops(&mut self, threshold: f32, nms_threshold: f32) {
        self.detection_ops = Some((threshold, nms_threshold));
    }

    pub fn render(&self, dot_path: String) -> Result<(), std::io::Error> {
        use tabbycat::attributes::*;
        use tabbycat::{AttrList, GraphBuilder, GraphType, Identity, StmtList, Edge};
        use std::fs::write;

        let names: Vec<String> = self.layers.iter().map(|x| {
            x.as_ref().borrow().name()
        }).collect();
        let n_layers = self.layers.len();

        let mut statements = StmtList::new();
        for i in 0..n_layers {
            let l_shape = self.layers[i].as_ref().borrow().shape().unwrap();
            statements = statements.add_node(
                Identity::id(&names[i]).unwrap(),
                None,
                Some(AttrList::new()
                    .add_pair(shape(Shape::None))
                    .add_pair(label(format!("<<table border='0' cellspacing='0'><tr><td border='1'>{}</td></tr><tr><td border='1'>{}</td></tr></table>>", names[i], l_shape)))
                )
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
        write(dot_path, graph.to_string().replace("\"", ""))?;
        Ok(())
    }

    fn set_batchsize(&mut self, batch: usize) -> Result<(), BuildError> {
        if let Some(_) = self.batchsize {
            return Err(BuildError::Rebuild(String::from("Batchsize already setted")))
        }
        self.batchsize = Some(batch);

        {
            let mut first_layer_ref = self.layers[0].borrow_mut();
            let input_l = first_layer_ref.as_any_mut().downcast_mut::<InputLayer>().unwrap();
            input_l.set_batchsize(batch);
        }
        
        for i in 1..self.layers.len() {
            let mut shapes = Vec::new();
            for (col, link) in self.adjency_matrix[i].iter().enumerate() {
                if *link == Link::Backward {
                    let shape = self.layers[col].borrow().shape().unwrap();
                    shapes.push(shape);
                }
            }
            self.layers[i].borrow_mut().infer_shape(shapes).map_err(|x| {
                BuildError::DimInferError(x)
            })?;
        }
        Ok(())
    }

    pub fn get_batchsize(&self) -> Option<usize> {
        self.batchsize.clone()
    }
    
    pub fn build_cudnn(&mut self, batchsize: usize, data_type: cudnnDataType) -> Result<(), BuildError> {
        if self.batchsize == None {
            return Err(BuildError::Runtime(RuntimeError::Other(String::from("Batchsize is not setted"))))
        }

        let engine = CUDNNEngine::new(data_type.clone(), self.batchsize.unwrap(), self.size)?;
    
        {   // Allocate tensors for first layer
            let mut first_layer = self.layers[0].borrow_mut();
            let info = Vec::new();
            first_layer.build(&engine, 0, true)?;
        }

        for i in 1..self.layers.len() {
            let has_depend_layers = self.adjency_matrix[i].iter().filter(|l| {
                l == &&Link::Forward
            }).count() > 1;
            let mut build_info = Vec::new();
            for (col, link) in self.adjency_matrix[i].iter().enumerate() {
                if *link == Link::Backward {
                    let info = self.layers[col].as_ref().borrow().get_build_information();
                    build_info.insert(0, info);
                }
            }
            self.layers[i].borrow_mut().build(context.clone(), data_type, build_info, has_depend_layers)?;
        }
        self.engine = Some(Box::new(engine));

        Ok(())
    }

    pub fn forward_debug(&mut self) -> Result<(), RuntimeError> {
        if self.batchsize == None {
            return Err(RuntimeError::Other(String::from("Batchsize is not setted")))
        }
        if self.context == None {
            return Err(RuntimeError::Other(String::from("Network not builded")))
        }

        for i in 0..self.layers.len() {
            let mut layer = self.layers[i].borrow_mut();
            layer.forward()?;
            cudaDeviceSynchronize().map_err(|e| {
                RuntimeError::Cuda(e)
            })?;

            let show = true;
            if show {
                let ptr = layer.get_build_information().tensor.borrow_mut().ptr();
                ptr.borrow().dump(&format!("./debug/activation/fused_{}.bin", layer.name()))?;

                let content: Vec<String> = ptr.borrow().download_with_conversion::<f32>()?[..20].iter().map(|x| {(*x).to_string()}).collect();
                println!("{} Data: [{}]", layer.name(), content.join(" "));
            }
        }
        cudaDeviceSynchronize().map_err(|e| {
            RuntimeError::Cuda(e)
        })?;

        Ok(())
    }

    pub fn forward(&mut self) -> Result<(), RuntimeError> {
        if self.batchsize == None {
            return Err(RuntimeError::Other(String::from("Batchsize is not setted")))
        }
        if self.context == None {
            return Err(RuntimeError::Other(String::from("Network not builded")))
        }

        for i in 0..self.layers.len() {
            self.layers[i].borrow_mut().forward()?;
        }
        cudaDeviceSynchronize().map_err(|e| {
            RuntimeError::Cuda(e)
        })?;

        Ok(())
    }

    pub fn load_darknet_weights(&mut self, path: &String) -> Result<(), BuildError> {
        use std::io::Read;
        use std::convert::TryInto;
        let mut file = std::fs::File::open(path).map_err(|e| {
            BuildError::Io(e)
        })?;
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer).map_err(|e| {
            BuildError::Io(e)
        })?;
        // _X variable means, that it not used, just for binary compatibility with darknet
        let major: i32 = i32::from_ne_bytes(buffer[0..4].try_into().unwrap());
        let minor: i32 = i32::from_ne_bytes(buffer[4..8].try_into().unwrap());
        let _revision: i32 = i32::from_ne_bytes(buffer[8..12].try_into().unwrap());
        let version = major * 10 + minor;
        
        let mut offset: usize;
        
        let _seen_images: usize;
        if version >= 2 {
            offset = 20;
            _seen_images = usize::from_ne_bytes(buffer[12..20].try_into().unwrap());
        } else {
            offset = 16;
            _seen_images = u32::from_ne_bytes(buffer[12..16].try_into().unwrap()) as usize;
        }
        
        for i in 1..self.layers.len() {
            offset = self.layers[i].borrow_mut().load_darknet_weights(offset, &buffer)?;
        }
        Ok(())

    }

    pub fn load_image(&mut self, image_path: String, batch_id: usize) -> Result<(), RuntimeError> {
        self.check_inited()?;
        if batch_id >= self.batchsize.unwrap() {
            return Err(RuntimeError::Other(String::from("Batch index is greater than network capacity")))
        }
        let mut layer = self.layers[0].borrow_mut();
        let input_layer = layer.as_any_mut().downcast_mut::<InputLayer>().unwrap();
        let shape = input_layer.shape().unwrap();
        let width = shape.W().unwrap();
        let height = shape.H().unwrap();

        let path = std::ffi::CString::new(image_path).unwrap();
        unsafe {
            let res = pnn_sys::load_image2batch(
                path.as_ptr(),
                batch_id,
                width, height,
                input_layer.get_input_tensor().unwrap().borrow_mut().ptr().borrow_mut().ptr() as *mut std::os::raw::c_void
            );
            if res == 0 {
                return Err(RuntimeError::Other(String::from("Couldnt load image")))
            }
        }
        Ok(())
    }

    pub fn load_bin(&self, bin_path: &String) -> Result<(), RuntimeError>  {
        self.check_inited()?;
        let mut layer = self.layers[0].borrow_mut();
        let input_layer = layer.as_any_mut().downcast_mut::<InputLayer>().unwrap();
        let ptr = input_layer.get_input_tensor().unwrap().borrow_mut().ptr();
        ptr.borrow_mut().load_bin(bin_path)?;
        Ok(())
    }

    pub fn get_detections(&self) -> Result<Vec<Vec<BoundingBox>>, RuntimeError> {
        self.check_inited()?;
        let batchsize = self.batchsize.unwrap();
        let (thresh, iou_tresh) = self.detection_ops.unwrap();
        
    }

    // #TODO: remove it
    pub fn get_input_ptr(&mut self) -> Result<*mut std::os::raw::c_void , BuildError> {
        self.check_inited().map_err(|e| {
            BuildError::Runtime(e)
        })?;
        let mut layer = self.layers[0].borrow_mut();
        let input_layer = layer.as_any_mut().downcast_mut::<InputLayer>().unwrap();

        Ok(input_layer.get_input_tensor().unwrap().borrow_mut().ptr().borrow_mut().ptr() as *mut std::os::raw::c_void)
    }

    pub fn get_size(&self) -> (usize, usize) {
        self.size
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn load_yolov4_csp() {
        let net = Network::from_darknet(&String::from("../cfgs/tests/yolov4-csp.cfg")).unwrap();
        assert_eq!(net.layers.len(), 161);
    }
}