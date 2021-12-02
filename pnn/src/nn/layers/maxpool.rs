use std::{
    collections::HashMap,
    self,
    any::Any,
    sync::atomic::{Ordering},
    rc::Rc,
    cell::RefCell
};

use crate::nn::shape::*;
use crate::nn::{Layer, LayerType, errors::*, BuildInformation};
use crate::parsers::{DeserializationError, parse_numerical_field, ensure_positive};
use crate::cudnn::{cudnnHandle_t, cudnnDataType, Tensor, DevicePtr};
use crate::nn::ops::{LayerOp, OutputTensor, PoolingOp};
use crate::nn::{CUDNNEngine, TRTBuilder};


//Maxpool
#[derive(Debug)]
pub struct MaxpoolLayer {
    // Layer name
    name: String,
    // Layer shape
    shape: Option<Rc<dyn Shape>>,
    // Window stride
    stride: usize,
    // Window sizeensure_positive
    size: usize,
    // Padding size
    padding: usize
}

const SUPPORTED_FIELDS: [&str; 3] = [
    "stride",
    "size",
    "padding",
];

impl Layer for MaxpoolLayer {
    fn name(&self) -> String {
        self.name.to_string()
    }

    fn shape(&self) -> Option<Rc<dyn Shape>> {
        self.shape.clone()
    }

    fn propose_name() -> String {
        static COUNTER: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        format!("Maxpool_{}", COUNTER.fetch_add(1, Ordering::Relaxed))
    }
 
    fn infer_shape(&mut self, input_shapes: Vec<Rc<dyn Shape>>) -> Result<(), ShapeError> {
        if input_shapes.len() != 1 {
            return Err(ShapeError(String::from("MaxpoolLayer must have exact one input layer")))
        }
        let input_shape = &input_shapes[0];
        if input_shape.dims().len() != 4 {
            return Err(ShapeError(String::from("MaxpoolLayer can be connected only with layer, which produce 4D Tensor with format NCHW")))
        }
        
        let h = (input_shape.H().unwrap() + self.padding - self.size) / self.stride + 1;
        if h < 1 {
            return Err(ShapeError(String::from("Resulting height is less then 1")));
        }

        let w = (input_shape.W().unwrap() + self.padding - self.size) / self.stride + 1;
        if w < 1 {
            return Err(ShapeError(String::from("Resulting width is less then 1")));
        }

        self.shape = Some(Rc::new(LayerShape::from_nchw(input_shape.N(), input_shape.C(), h, w)));
        Ok(())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn from_config(config: HashMap<String, String>) -> Result<Box<dyn Layer>, DeserializationError> {
        let proposed_name = MaxpoolLayer::propose_name();
        let name: String = config.get("name").unwrap_or(&proposed_name).to_string();
        let shape = None;
        let stride = parse_numerical_field::<usize>(&config, "stride", true, None)?.unwrap();
        ensure_positive(stride, "stride", "MaxpoolLayer")?;
        let size = parse_numerical_field::<usize>(&config, "size", false, Some(stride))?.unwrap();
        ensure_positive(stride, "size", "MaxpoolLayer")?;
        let padding = parse_numerical_field::<usize>(&config, "padding", false, Some(size - 1))?.unwrap();
        
        let _ = config.keys().filter(|k| {
            !SUPPORTED_FIELDS.contains(&&k[..])
        }).map(|k| {
            log::warn!("Not supported darknet field during deserialization of '{}'. Field '{}' not recognized", name, k)
        });

        Ok(Box::new(MaxpoolLayer{name, shape, stride, size, padding}))
    }


    fn ltype(&self) -> LayerType {
        LayerType::Maxpool
    }

    fn build_cudnn(&mut self, 
        engine: Rc<RefCell<CUDNNEngine>>,
        indeces: Vec<usize>,
        has_depend_layers: bool
    ) -> Result<(), BuildError> {
        let reusable = !has_depend_layers;
        let info: Vec<BuildInformation> = indeces.iter().map(|x| {
            engine.borrow().get_info(*x)
        }).collect();
        let mut tensor;
        let data_type = engine.borrow().dtype();
        let mut operations: Vec<Box<dyn LayerOp>> = Vec::new();

        let shape = self.shape().unwrap();
        let input_tensor = info[0].tensor.clone();
        if shape.as_ref().dims() == input_tensor.borrow().shape().dims() && info[0].reusable {
            tensor = input_tensor.clone()
        } else {
            let ptr = Rc::new(RefCell::new(
                DevicePtr::new(data_type.clone(), shape.size()).map_err(|e| {
                    BuildError::Runtime(e)
                })?
            ));
            let tensor_shape: Box<dyn Shape> = Box::new(LayerShape::new(shape.dims()));
            tensor = Rc::new(RefCell::new(
                Tensor::new(tensor_shape, ptr).map_err(|e| {
                    BuildError::Runtime(e)
                })?
            ));
        }

        operations.push(
            Box::new(PoolingOp::new(
                engine.borrow().context(),
                input_tensor.clone(),
                tensor.clone(),
                &data_type, 
                true,
                self.stride, self.stride,
                self.padding / 2, self.padding / 2,
                self.size, self.size
            ).map_err(|e| {
                BuildError::Runtime(e)
            })?)
        );
        engine.borrow_mut().add_layer(operations, BuildInformation{tensor, reusable});
        Ok(())
    }

    fn build_trt(&mut self, 
        engine: Rc<RefCell<TRTBuilder>>,
        indeces: Vec<usize>
    ) -> Result<(), BuildError> {
        let mut engine = engine.borrow_mut();
        let mut id: usize = engine.last_op_id(indeces[0]);
        id = engine.add_pooling(
            id,
            self.stride,
            self.size,
            self.padding,
            true
        )?;
        engine.finilize_layer(id);
        Ok(())
    }
}


impl MaxpoolLayer {
}

#[cfg(test)]
mod tests {
    use super::*;

    fn generate_config() -> HashMap<String, String> {
        let mut config: HashMap<String, String> = HashMap::new();
        config.insert(String::from("stride"), String::from("3"));
        config
    }

    #[test]
    fn test_create_minimal() {
        let layer = MaxpoolLayer::from_config(generate_config()).unwrap();
        let layer = layer.as_any().downcast_ref::<MaxpoolLayer>().unwrap();

        assert_eq!(layer.stride, 3);
        assert_eq!(layer.size, 3);
    }


    #[test]
    fn test_ltype() {
        let layer = MaxpoolLayer::from_config(generate_config()).unwrap();
        assert_eq!(layer.ltype(), LayerType::Maxpool);
    }

    #[test]
    fn test_create_sized() {
        let mut config: HashMap<String, String> = HashMap::new();
        config.insert(String::from("stride"), String::from("5"));
        config.insert(String::from("size"), String::from("9"));

        let layer = MaxpoolLayer::from_config(config).unwrap();
        let layer = layer.as_any().downcast_ref::<MaxpoolLayer>().unwrap();
        assert_eq!(layer.stride, 5);
        assert_eq!(layer.size, 9);
    }

    #[test]
    #[should_panic(expected = "Key 'stride' is mandatory")]
    fn test_create_fail_parse_float() {
        let mut config: HashMap<String, String> = HashMap::new();
        config.insert(String::from("size"), String::from("2"));
        MaxpoolLayer::from_config(config).unwrap();
    }
    #[test]
    fn test_infer_shape_simple() {
        let shapes: Vec<Rc<dyn Shape>> = vec![Rc::new(LayerShape::from_nchw(32, 3, 128, 128))];
        let mut config: HashMap<String, String> = HashMap::new();
        config.insert(String::from("stride"), String::from("2"));


        let mut layer = MaxpoolLayer::from_config(config).unwrap();
        layer.infer_shape(shapes).unwrap();
        let conv_layer = layer.as_any().downcast_ref::<MaxpoolLayer>().unwrap();

        assert_eq!(*conv_layer.shape().unwrap().dims(), vec![32, 3, 64, 64]);
    }

    #[test]
    fn test_infer_shape_sized() {
        let mut config = generate_config();
        config.insert(String::from("size"), String::from("8"));
        config.insert(String::from("stride"), String::from("4"));
        let shapes: Vec<Rc<dyn Shape>> = vec![Rc::new(LayerShape::from_nchw(32, 3, 128, 100))];


        let mut layer = MaxpoolLayer::from_config(config).unwrap();
        layer.infer_shape(shapes).unwrap();
        let conv_layer = layer.as_any().downcast_ref::<MaxpoolLayer>().unwrap();

        assert_eq!(*conv_layer.shape().unwrap().dims(), vec![32, 3, 32, 25]);
    }
    #[test]
    #[should_panic(expected = "MaxpoolLayer must have exact one input layer")]
    fn test_infer_shape_invalid_count() {
        let shapes: Vec<Rc<dyn Shape>> = vec![
            Rc::new(LayerShape::from_nchw(32, 3, 128, 100)),
            Rc::new(LayerShape::from_nchw(32, 3, 128, 100))
            ];

        let mut layer = MaxpoolLayer::from_config(generate_config()).unwrap();
        layer.infer_shape(shapes).unwrap();
    }

    #[test]
    #[should_panic(expected = "MaxpoolLayer can be connected only with layer, which produce 4D Tensor with format NCHW")]
    fn test_infer_shape_3d() {
        let shapes: Vec<Rc<dyn Shape>> = vec![
            Rc::new(LayerShape::from_nch(32, 3, 128))
            ];

        let mut layer = MaxpoolLayer::from_config(generate_config()).unwrap();
        layer.infer_shape(shapes).unwrap();
    }

}