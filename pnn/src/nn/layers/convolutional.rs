use std::{
    collections::HashMap,
    self,
    any::Any,
    sync::atomic::{Ordering},
    rc::Rc,
    convert::TryFrom,
    cell::RefCell
};

use crate::nn::shape::*;
use crate::nn::{Layer, LayerType, errors::*, ActivationType, BuildInformation};
use crate::parsers::{DeserializationError, parse_numerical_field};
use crate::cudnn::{cudnnHandle_t, cudnnDataType, Tensor, DevicePtr};
use crate::nn::ops::{LayerOp, OutputTensor, ConvolutionOp, BatchnormOp, ActivationOp};

type F32Vec = Vec<f32>;

#[derive(Debug)]
struct Weights {
    biases: F32Vec,
    batchnorm: Option<(F32Vec, F32Vec, F32Vec)>,
    conv: F32Vec
}

//Convolution
#[derive(Debug)]
pub struct ConvolutionalLayer {
    // Layer name
    name: String,
    // Layer shape
    shape: Option<Rc<dyn Shape>>,
    // Filters count
    filters: usize,
    // Perform batch normalization before inference
    batch_normalize: bool,
    // Window size, f.e. size=3 means to use 3x3 conv
    size: usize,
    // Convolution stride
    stride: usize,
    // Add pads to input tensor to keep spacial dims
    pad: bool,
    // Paddings size
    padding: usize,
    // Activation
    activation: ActivationType,
    // List of operations
    operations: Vec<Box<dyn LayerOp>>,
    // Can be reusable
    reusable: bool,
    // Output tensor
    tensor: Option<OutputTensor>,
    // Weights
    weights: Option<Weights>,
    // previous channel size
    prev_c: Option<usize>

}

const SUPPORTED_FIELDS: [&str; 7] = [
    "filters",
    "batch_normalize",
    "size",
    "stride",
    "pad",
    "padding",
    "activation"
];

impl Layer for ConvolutionalLayer {
    fn name(&self) -> String {
        self.name.to_string()
    }

    fn shape(&self) -> Option<Rc<dyn Shape>> {
        self.shape.clone()
    }

    fn propose_name() -> String {
        static COUNTER: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        format!("Convolution_{}", COUNTER.fetch_add(1, Ordering::Relaxed))
    }

    fn infer_shape(&mut self, input_shapes: Vec<Rc<dyn Shape>>) -> Result<(), ShapeError> {
        if input_shapes.len() != 1 {
            return Err(ShapeError(String::from("Convolutional layer must have exact one input layer")))
        }
        let input_shape = &input_shapes[0];
        if input_shape.dims().len() != 4 {
            return Err(ShapeError(String::from("Convolutional layer can be connected only with layer, which produce 4D Tensor with format NCHW")))
        }
        let n = input_shape.N();
        let delta = if self.pad {self.padding as i32} else {0};

        let h: i32 = (input_shape.H().unwrap() as i32
                      - self.size as i32
                      + 2 * delta
                      + self.stride as i32) / self.stride as i32;
        if h <= 0 {
            return Err(ShapeError(format!("Couldnt set height to {}", h)));
        }

        let w: i32 = (input_shape.W().unwrap() as i32
                      - self.size as i32
                      + 2 * delta
                      + self.stride as i32) / self.stride as i32;
        if w <= 0 {
            return Err(ShapeError(format!("Couldnt set width to {}", w)));
        }

        self.shape = Some(Rc::new(LayerShape::from_nchw(n, self.filters, h as usize, w as usize)));
        self.prev_c = Some(input_shape.C());
        Ok(())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn from_config(config: HashMap<String, String>) -> Result<Box<dyn Layer>, DeserializationError> {
        let proposed_name = ConvolutionalLayer::propose_name();
        let name: String = config.get("name").unwrap_or(&proposed_name).to_string();
        let shape = None;
        let filters = parse_numerical_field::<usize>(&config, "filters", true, None)?.unwrap();
        let batch_normalize = parse_numerical_field::<usize>(&config, "batch_normalize", false, Some(0))?.unwrap() != 0;
        let size = parse_numerical_field::<usize>(&config, "size", false, Some(1))?.unwrap();
        let stride = parse_numerical_field::<usize>(&config, "stride", false, Some(1))?.unwrap();

        let pad = parse_numerical_field::<usize>(&config, "pad", false, Some(0))?.unwrap() != 0;
        let default_padding = if pad {size / 2} else {0};
        let padding = parse_numerical_field::<usize>(&config, "padding", false, Some(default_padding))?.unwrap();
        let activation = ActivationType::try_from(&config.get("activation").unwrap_or(&String::from("linear")).to_string())?;
        
        let _ = config.keys().filter(|k| {
            !SUPPORTED_FIELDS.contains(&&k[..])
        }).map(|k| {
            log::warn!("Not supported darknet field during deserialization of '{}'. Field '{}' not recognized", name, k)
        });

        let tensor = None;
        let operations = vec![];
        let reusable = false;
        let weights = None;
        let prev_c = None;

        Ok(Box::new(ConvolutionalLayer{name, shape, 
            filters, batch_normalize, 
            size, stride,
            pad, padding, 
            activation, operations,
            reusable, tensor,
            weights, prev_c
        }))
    }

    fn layer_type(&self) -> LayerType {
        LayerType::Convolutional
    }

    fn get_build_information(&self) -> BuildInformation {
        BuildInformation{tensor: self.tensor.as_ref().unwrap().clone(), reusable: self.reusable}
    }

    fn get_operations(&mut self) -> &mut Vec<Box<dyn LayerOp>> {
        &mut self.operations
    }

    fn build(&mut self, 
        context: Rc<cudnnHandle_t>,
        data_type: cudnnDataType,
        info: Vec<BuildInformation>,
        has_depend_layers: bool
    ) -> Result<(), BuildError> {
        self.reusable = !has_depend_layers;

        let shape = self.shape().unwrap();
        let input_tensor = info[0].tensor.clone();
        // Inplace conv not work ;(
        if shape.as_ref().dims() == input_tensor.borrow().shape().dims() && info[0].reusable && false {
            self.tensor = Some(input_tensor.clone())
        } else {
            let ptr = Rc::new(RefCell::new(
                DevicePtr::new(data_type.clone(), shape.size()).map_err(|e| {
                    BuildError::Runtime(e)
                })?
            ));
            let tensor_shape: Box<dyn Shape> = Box::new(LayerShape::new(shape.dims()));
            let tensor = Rc::new(RefCell::new(
                Tensor::new(tensor_shape, ptr).map_err(|e| {
                    BuildError::Runtime(e)
                })?
            ));

            self.tensor = Some(tensor);
        }
        let t = self.tensor.as_ref().unwrap();
        let conv_weights = match &self.weights {
            Some(w) => Some(&w.conv),
            _ => None
        };
        self.operations.push(
            Box::new(ConvolutionOp::new(
                context.clone(),
                input_tensor.clone(),
                t.clone(),
                &data_type, 
                self.filters,
                self.prev_c.unwrap(),
                self.size, self.size,
                self.padding, self.padding,
                self.stride, self.stride,
                conv_weights
            ).map_err(|e| {
                BuildError::Runtime(e)
            })?)
        );

        if self.batch_normalize {
            let biases = &self.weights.as_ref().unwrap().biases;
            let batch_weights = &self.weights.as_ref().unwrap().batchnorm.as_ref().unwrap();

            let batch_weights = Some((biases, &batch_weights.0, &batch_weights.1, &batch_weights.2));
            self.operations.push(
                Box::new(BatchnormOp::new(
                    context.clone(),
                    t.clone(),
                    t.clone(),
                    &data_type, 
                    self.filters,
                    batch_weights
                ).map_err(|e| {
                    BuildError::Runtime(e)
                })?)
            );
        }

        self.operations.push(
            Box::new(ActivationOp::new(
                context.clone(), 
                t.clone(),
                &data_type, 
                &self.activation
            ).map_err(|e| {
                BuildError::Runtime(e)
            })?)
        );
        Ok(())
    }

    // Initialize weights using darknet model file. Consume initial offset and return new
    fn load_darknet_weights(&mut self, offset: usize, bytes: &Vec<u8>) -> Result<usize, BuildError> {
        use crate::parsers::load_f32_vec;
        let mut batchnorm = None;

        let (biases, mut inner_offset) = load_f32_vec(offset, bytes, self.filters)?;
        if self.batch_normalize {
            let (scales, offset) = load_f32_vec(inner_offset, bytes, self.filters)?;
            let (rolling_mean, offset) = load_f32_vec(offset, bytes, self.filters)?;
            let (rolling_variance, new_offset) = load_f32_vec(offset, bytes, self.filters)?;
            inner_offset = new_offset;
            batchnorm = Some((scales, rolling_mean, rolling_variance));
        }
        let (conv, offset) = load_f32_vec(inner_offset, bytes, self.filters * self.size * self.size * self.prev_c.unwrap())?;
        self.weights = Some(Weights{batchnorm, biases, conv});

        Ok(offset)
    }
}


impl ConvolutionalLayer {
}

#[cfg(test)]
mod tests {
    use super::*;

    fn generate_config() -> HashMap<String, String> {
        let mut config: HashMap<String, String> = HashMap::new();
        config.insert(String::from("filters"), String::from("32"));
        config
    }

    #[test]
    fn test_create_minimal() {
        let layer = ConvolutionalLayer::from_config(generate_config()).unwrap();
        let conv_layer = layer.as_any().downcast_ref::<ConvolutionalLayer>().unwrap();
        assert_eq!(conv_layer.filters, 32);
        assert_eq!(conv_layer.batch_normalize, false);
        assert_eq!(conv_layer.size, 1);
        assert_eq!(conv_layer.stride, 1);
        assert_eq!(conv_layer.pad, false);
        assert_eq!(conv_layer.padding, 0);
        assert_eq!(conv_layer.activation, ActivationType::Linear);
    }

    #[test]
    fn test_create_with_pad() {
        let mut config = generate_config();
        config.insert(String::from("pad"), String::from("1"));
        config.insert(String::from("size"), String::from("5"));
        config.insert(String::from("activation"), String::from("mish"));

        let layer = ConvolutionalLayer::from_config(config).unwrap();
        let conv_layer = layer.as_any().downcast_ref::<ConvolutionalLayer>().unwrap();
        assert_eq!(conv_layer.filters, 32);
        assert_eq!(conv_layer.batch_normalize, false);
        assert_eq!(conv_layer.size, 5);
        assert_eq!(conv_layer.stride, 1);
        assert_eq!(conv_layer.pad, true);
        assert_eq!(conv_layer.padding, 2);
        assert_eq!(conv_layer.activation, ActivationType::Mish);
    }

    #[test]
    fn test_infer_shape_simple() {
        let shapes: Vec<Rc<dyn Shape>> = vec![Rc::new(LayerShape::from_nchw(32, 3, 128, 128))];


        let mut layer = ConvolutionalLayer::from_config(generate_config()).unwrap();
        layer.infer_shape(shapes).unwrap();
        let conv_layer = layer.as_any().downcast_ref::<ConvolutionalLayer>().unwrap();

        assert_eq!(*conv_layer.shape().unwrap().dims(), vec![32, 32, 128, 128]);
    }

    #[test]
    fn test_infer_shape_sized() {
        let mut config = generate_config();
        config.insert(String::from("size"), String::from("5"));
        let shapes: Vec<Rc<dyn Shape>> = vec![Rc::new(LayerShape::from_nchw(32, 3, 128, 100))];


        let mut layer = ConvolutionalLayer::from_config(config).unwrap();
        layer.infer_shape(shapes).unwrap();
        let conv_layer = layer.as_any().downcast_ref::<ConvolutionalLayer>().unwrap();

        assert_eq!(*conv_layer.shape().unwrap().dims(), vec![32, 32, 124, 96]);
    }

    #[test]
    fn test_infer_shape_sized_padded() {
        let mut config = generate_config();
        config.insert(String::from("size"), String::from("5"));
        config.insert(String::from("pad"), String::from("1"));
        let shapes: Vec<Rc<dyn Shape>> = vec![Rc::new(LayerShape::from_nchw(32, 3, 128, 100))];


        let mut layer = ConvolutionalLayer::from_config(config).unwrap();
        layer.infer_shape(shapes).unwrap();
        let conv_layer = layer.as_any().downcast_ref::<ConvolutionalLayer>().unwrap();

        assert_eq!(*conv_layer.shape().unwrap().dims(), vec![32, 32, 128, 100]);
    }

    #[test]
    fn test_infer_shape_sized_padded_strided() {
        let mut config = generate_config();
        config.insert(String::from("size"), String::from("5"));
        config.insert(String::from("pad"), String::from("1"));
        config.insert(String::from("stride"), String::from("2"));
        let shapes: Vec<Rc<dyn Shape>> = vec![Rc::new(LayerShape::from_nchw(32, 3, 128, 100))];


        let mut layer = ConvolutionalLayer::from_config(config).unwrap();
        layer.infer_shape(shapes).unwrap();
        let conv_layer = layer.as_any().downcast_ref::<ConvolutionalLayer>().unwrap();

        assert_eq!(*conv_layer.shape().unwrap().dims(), vec![32, 32, 64, 50]);
    }

    #[test]
    #[should_panic(expected = "Convolutional layer must have exact one input layer")]
    fn test_infer_shape_invalid_count() {
        let shapes: Vec<Rc<dyn Shape>> = vec![
            Rc::new(LayerShape::from_nchw(32, 3, 128, 100)),
            Rc::new(LayerShape::from_nchw(32, 3, 128, 100))
            ];

        let mut layer = ConvolutionalLayer::from_config(generate_config()).unwrap();
        layer.infer_shape(shapes).unwrap();
    }

    #[test]
    #[should_panic(expected = "Convolutional layer can be connected only with layer, which produce 4D Tensor with format NCHW")]
    fn test_infer_shape_3d() {
        let shapes: Vec<Rc<dyn Shape>> = vec![
            Rc::new(LayerShape::from_nch(32, 3, 128))
            ];

        let mut layer = ConvolutionalLayer::from_config(generate_config()).unwrap();
        layer.infer_shape(shapes).unwrap();
    }

    #[test]
    fn test_layer_type() {
        let layer = ConvolutionalLayer::from_config(generate_config()).unwrap();
        assert_eq!(layer.layer_type(), LayerType::Convolutional);
    }

}