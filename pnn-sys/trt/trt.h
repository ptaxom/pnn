#pragma once

#include <NvInfer.h>
#include <cudnn.h>

extern "C" {
    void* builder_create(cudnnDataType_t dataType, size_t maxBatchsize);

    void  builder_destroy(void* builder);

    int   builder_add_convolution(void* builder,
        size_t input_id, 
        size_t feature_maps,
        size_t input_c, 
        size_t kernel_size, 
        size_t padding, 
        size_t stride,
        float* kernel,
        float* biases
    );

    int   builder_add_activation(void* builder, size_t input_id, const char* act_name);

    int   builder_add_shortcut(void* builder, size_t n_layers, size_t *layers);

    int   builder_add_upsample(void* builder, size_t input_id, size_t stride);

    int   builder_add_input(void* builder, const char* name, size_t channels, size_t width, size_t height);

    void  builder_add_yolo(void* builder, size_t input_id, const char* name);

    int   builder_add_route(void* builder, size_t n_layers, size_t *layers);

    int   builder_add_pooling(void* builder, size_t input_id, size_t stride, size_t window_size, size_t padding, size_t is_max);

    int   builder_build(void* builder, size_t avgIters, size_t minIters, const char* engine_path);

}