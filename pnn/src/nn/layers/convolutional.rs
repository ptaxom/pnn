use std::{
    collections::HashMap,
    self,
    any::Any,
    sync::atomic::{Ordering},
    rc::Rc
};

use crate::nn::shape::*;
use crate::nn::Layer;
use crate::parsers::{DeserializationError, parse_numerical_field};


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
    // Activation. #TODO: add support of activation :)
    activation: String
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
            return Err(ShapeError{description: String::from("Convolutional layer must have exact one input layer")})
        }
        let input_shape = &input_shapes[0];
        if input_shape.dims().len() != 4 {
            return Err(ShapeError{description: String::from("Convolutional layer can be connected only with layer, which produce 4D Tensor with format NCHW")})
        }
        let n = input_shape.N();
        let delta = if self.pad {self.padding as i32} else {0};

        let h: i32 = (input_shape.H().unwrap() as i32
                      - self.size as i32
                      + 2 * delta
                      + self.stride as i32) / self.stride as i32;
        if h <= 0 {
            return Err(ShapeError{description: format!("Couldnt set height to {}", h)});
        }

        let w: i32 = (input_shape.W().unwrap() as i32
                      - self.size as i32
                      + 2 * delta
                      + self.stride as i32) / self.stride as i32;
        if w <= 0 {
            return Err(ShapeError{description: format!("Couldnt set width to {}", w)});
        }

        self.shape = Some(Rc::new(LayerShape::from_nchw(n, self.filters, h as usize, w as usize)));
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
        let activation: String = config.get("activation").unwrap_or(&String::from("linear")).to_string();
        
        let _ = config.keys().filter(|k| {
            !SUPPORTED_FIELDS.contains(&&k[..])
        }).map(|k| {
            log::warn!("Not supported darknet field during deserialization of '{}'. Field '{}' not recognized", name, k)
        });

        Ok(Box::new(ConvolutionalLayer{name, shape, filters, batch_normalize, size, stride, pad, padding, activation}))
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
        assert_eq!(conv_layer.activation, "linear");
    }

    #[test]
    fn test_create_with_pad() {
        let mut config = generate_config();
        config.insert(String::from("pad"), String::from("1"));
        config.insert(String::from("size"), String::from("5"));

        let layer = ConvolutionalLayer::from_config(config).unwrap();
        let conv_layer = layer.as_any().downcast_ref::<ConvolutionalLayer>().unwrap();
        assert_eq!(conv_layer.filters, 32);
        assert_eq!(conv_layer.batch_normalize, false);
        assert_eq!(conv_layer.size, 5);
        assert_eq!(conv_layer.stride, 1);
        assert_eq!(conv_layer.pad, true);
        assert_eq!(conv_layer.padding, 2);
        assert_eq!(conv_layer.activation, "linear");
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

}