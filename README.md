# pnn

pnn is [Darknet](https://github.com/alexeyAB/darknet) compatible neural nets inference engine implemented in Rust. By optimizing  was achieved significant performance increment(especially in FP16 mode). _pnn_ provide CUDNN-based and TensorRT-based inference engines.
<center>
<img src="assets/result.gif">
</center>

## FPS Performance
Performance is measured at RTX 3070Ti, TensorRT v8.2.1, CUDNN v8.3.0, NVCC/CUDA Runtime 11.5, SM=80. For fair comparison was used [tkDNN](https://github.com/ceccocats/tkDNN) from _tensorrt8_ branch. 
-  **YOLOv4 CSP 512x512**
    | Configuration | Darknet    |  tkDNN         | pnn + CUDNN   | pnn + TensorRT |
    | :------:      | :-----:    | :-----:        | :-----:       | :-----:        |
    | BS=1, FP32    |  87.8      | 98.2(112.9**)  | 98.1(107.0**) | __108.7(119.6**)__
    | BS=1, FP16    |  99.9*     | __221.2(359.0**)__ | 159(183.7**)  | 197.3(238.0**)
    | BS=4, FP32    |    -       | 121.0(129.3**) | 117.4(517**)  | __130.1(590.0**)__
    | BS=4, FP16    |    -       | 268.3(493.4**) | 193.2(869.0**)| __230.7(1150.5**)__
- <details>
  <summary><b>YOLOv4 416x416[WIP]</b></summary>
  <br>
  <table border=1>
    <tr>
        <td>Configuration</td>
        <td>Darknet</td>
        <td>tkDNN</td>
        <td>pnn(CUDNN)</td>
        <td>pnn(TensorRT)</td>        
    </tr>
    <tr>
        <td>BS=1, FP32</td>
        <td>60.3</td>
        <td>121.7(133.8*)</td>
        <td>N/A</td>
        <td>N/A</td>        
    </tr>
        <tr>
        <td>BS=1, FP16</td>
        <td>69.7</td>
        <td>290.5(455.1*)</td>
        <td>N/A</td>
        <td>N/A</td>        
    </tr>
    </tr>
        <tr>
        <td>BS=4, FP32</td>
        <td>N/A</td>
        <td>161.4(179.8*)</td>
        <td>N/A</td>
        <td>N/A</td>        
    </tr>
    </tr>
    <tr>
        <td>BS=4, FP16</td>
        <td>N/A</td>
        <td>365.1(632.1*)</td>
        <td>N/A</td>
        <td>N/A</td>        
    </tr>
  </table>
  </details>
\* - Actually, Darknet hasnt FP16 mode, it operate in mixed precision

\*\* - Main value is full inference time, including reading, preprocessing and postprocessing. Value in brackets is clear inference time. During benchmark nor of Darknet, tkDNN or pnn doesnt render video to screen/file. If perform benchmark with render you will get 3-5% decreasing for pnn/Darknet with multithreaded loader/renderer and ~30% for tkDNN with single threaded renderer.
## Usage
- To build TensorRT engine use
    ```bash
    $ ./pnn build --help    
    pnn-build 

    Build TensorRT engine file

    USAGE:
        pnn build [OPTIONS] --weights <WEIGHTS> --config <CONFIG>

    OPTIONS:
        -b, --batchsize <BATCHSIZE>    Batchsize. [default: 1]
        -c, --config <CONFIG>          Path to config
        -h, --help                     Print help information
            --half                     Build HALF precision engine
        -o, --output <OUTPUT>          Output engine
        -w, --weights <WEIGHTS>        Path to weights
    # For example
    $ ./pnn build -b 4 -c ../../cfgs/tests/yolov4-csp.cfg -w ../../../models/yolov4-csp.weights -o ../../yolo_fp16_bs4.engine --half
    ```
- To run/benchmark/render use
    ```bash
    ./pnn benchmark --help

    Do performance benchmark

    USAGE:
        pnn benchmark [OPTIONS] --weights <WEIGHTS> --config <CONFIG> --input <INPUT>

    OPTIONS:
        -b, --batchsize <BATCHSIZE>          Batchsize [default: 1]
        -c, --config <CONFIG>                Path to config
            --classes-file <CLASSES_FILE>    Confidence threshold [default: ./cfgs/tests/coco.names]
        -h, --help                           Print help information
            --half                           Build HALF precision engine
        -i, --input <INPUT>                  Input file
            --iou-tresh <IOU_TRESH>          Confidence threshold [default: 0.45]
        -o, --output <OUTPUT>                Output render file
        -s, --show                           Render window during work
            --threshold <THRESHOLD>          Confidence threshold [default: 0.45]
            --trt                            Load as TensorRT engine
        -w, --weights <WEIGHTS>              Path to weights
    # For example
    $ ./pnn benchmark -w ~/Sources/models/yolov4-p6.weights -c ~/Sources/models/yolov4-p6.cfg -s -b 1 -i ~/Sources/models/yolo_test.mp4 # Run yolov4-p6 with darknet FP32 and BS1 engine and render result to screen

    $ ./pnn benchmark --trt --weights yolo_fp16_bs4.engine -c cfgs/tests/yolov4-csp.cfg --input ../models/yolo_test.mp4 --output res.avi # Run yolo_fp16_bs4.engine engine with predefined in build-time settings and save to result to res.avi
    ```
    The result would be like this one
    ```
    Stats for      ../models/yolo_test.mp4
    Data type:     FP16
    Batchsize:     1
    Total frames:  1213
    FPS:           147.48 # END-TO-END FPS, including reading/preprocessing/rendering time
    INF+NMS FPS:   159.13 # Inference time + post processing FPS
    Inference FPS: 173.68 # Only inference measured. In case bs != 1 counted by bs * inference FPS
    ```
- To show model architecture use
    ```bash
    $ ./pnn dot --help
    Build dot graph of model

    USAGE:
        pnn dot --config <CONFIG> --output <OUTPUT>

    OPTIONS:
        -c, --config <CONFIG>    Path to config
        -h, --help               Print help information
        -o, --output <OUTPUT>    Output dot file
    # For further conversion use 
    $ dot -Tpng path.dot > path.png
    ```
    <details>
    <summary>YOLOv4 CSP architecture</summary>
    <br>
    <img src="assets/yolov4-csp.png">
    </details>


## Requirements
- Rust 2021 edition
- Clang ≥ 13.0
- GCC ≥ 9.0
- NVCC ≥ 10
- CUDNN ≥ 8
- TensorRT ≥ 8
- OpenCV ≥ 4.4

## Roadmap
- [x] CUDNN FP32 Support
- [x] dot files render
- [x] TensorRT FP32 Support
- [x] FP16 mode
- [ ] Python bindings
- [ ] Releases & packages for Rust/C++/Python
- [ ] INT8 support
- [ ] Refitting engine