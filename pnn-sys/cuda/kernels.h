#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cudnn.h>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

const size_t BLOCK_SIZE = 512;

struct BoundingBox{
    float x0;
    float y0;
    float x1;
    float y1;
    size_t class_id;
    float probability;
    float objectness;
};

struct InferStats {
    double inference_time;
    double inference_with_nms;
    double total_time;
    size_t total_frames;  
};

std::vector<std::string> load_classes(const char** c_classes);
std::vector<BoundingBox> load_bboxes(size_t n_boxes, BoundingBox* const boxes);
void draw_bboxes(cv::Mat &image, const std::vector<BoundingBox> &bboxes, const std::vector<std::string> &classes);

extern "C" {
    dim3 get_gridsize(size_t elements);

    cudaError_t activation_mish_fp16(void* data, size_t elements, cudaStream_t stream);
    cudaError_t activation_mish_fp32(void* data, size_t elements, cudaStream_t stream);
    cudaError_t activation_mish_fp64(void* data, size_t elements, cudaStream_t stream);

    cudaError_t activation_leaky_fp16(void* data, size_t elements, cudaStream_t stream);
    cudaError_t activation_leaky_fp32(void* data, size_t elements, cudaStream_t stream);
    cudaError_t activation_leaky_fp64(void* data, size_t elements, cudaStream_t stream);

    cudaError_t upsample_forward_fp16(void* input, size_t n, size_t c, size_t h, size_t w, size_t stride, float scale, void* output, cudaStream_t stream);
    cudaError_t upsample_forward_fp32(void* input, size_t n, size_t c, size_t h, size_t w, size_t stride, float scale, void* output, cudaStream_t stream);
    cudaError_t upsample_forward_fp64(void* input, size_t n, size_t c, size_t h, size_t w, size_t stride, float scale, void* output, cudaStream_t stream);

    cudaError_t cvt_ptr_data(void* output, void* input, size_t n_elements, size_t otype, size_t itype, cudaStream_t stream);

    cudaError_t add_bias(void *inplace_data, void* biases, size_t n_elements, size_t n_channels, size_t channel_size, cudnnDataType_t dtype, cudaStream_t stream);

    int load_image2batch(const char* image_path, size_t batch_id, int width, int height, void* input_data);
    int render_bboxes(const char* image_path, size_t n_boxes, void* const boxes, const char** classes, const char* window_name);

    InferStats visual_demo(const char* video_path, 
                 const char** c_classes, 
                 size_t batchsize,
                 size_t width,
                 size_t height,
                 void* inp_ptr,
                 void* model_ptr,
                 BoundingBox* (*infer_call)(void* model, size_t *n_boxes, double *infer_time),
                 const char* output_path,
                 bool show
                 );

}