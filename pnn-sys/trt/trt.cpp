#include "trt.h"

#include "builder.hpp"
#include "engine.hpp"

void* builder_create(cudnnDataType_t dataType, size_t maxBatchsize) {
    return new TRTBuilder(dataType, maxBatchsize);
}

void builder_destroy(void* builder) {
    delete static_cast<TRTBuilder*>(builder);
}


int builder_add_convolution(void* builder,
    size_t input_id,
    size_t feature_maps,
    size_t input_c,
    size_t kernel_size,
    size_t padding,
    size_t stride,
    float* kernel, 
    float* biases
    ) {
    int64_t params_count = static_cast<int64_t>(feature_maps * input_c * kernel_size * kernel_size);
    Weights wKernel{nvinfer1::DataType::kFLOAT, kernel, params_count};
    Weights wBiases{nvinfer1::DataType::kFLOAT, biases, static_cast<int64_t>(feature_maps)};
    return static_cast<TRTBuilder*>(builder)->addConvolution(input_id, feature_maps, kernel_size, padding, stride, wKernel, wBiases);
}

int builder_add_activation(void* builder, size_t input_id, const char* act_name) {
    return static_cast<TRTBuilder*>(builder)->addActivation(input_id, act_name);
}

int builder_add_shortcut(void* builder, size_t n_layers, size_t *layers) {
    std::vector<size_t> indeces;
    for(size_t i = 0; i < n_layers; i++) indeces.push_back(layers[i]);
    return static_cast<TRTBuilder*>(builder)->addShortcut(indeces);
}

int builder_add_upsample(void* builder, size_t input_id, size_t stride) {
    return static_cast<TRTBuilder*>(builder)->addUpsample(input_id, stride);
}

int builder_add_input(void* builder, const char* name, size_t channels, size_t width, size_t height) {
    return static_cast<TRTBuilder*>(builder)->addInput(name, channels, height, width);
}

void builder_add_yolo(void* builder, size_t input_id, const char* name) {
    static_cast<TRTBuilder*>(builder)->addYolo(input_id, name);
}

int builder_add_route(void* builder, size_t n_layers, size_t *layers) {
    std::vector<size_t> indeces;
    for(size_t i = 0; i < n_layers; i++) indeces.push_back(layers[i]);
    return static_cast<TRTBuilder*>(builder)->addRoute(indeces);
}

int builder_add_pooling(void* builder, size_t input_id, size_t stride, size_t window_size, size_t padding, size_t is_max) {
    return static_cast<TRTBuilder*>(builder)->addPooling(input_id, stride, window_size, padding, is_max);
}

int builder_build(void* builder, size_t avgIters, size_t minIters, const char* engine_path) {
    return static_cast<TRTBuilder*>(builder)->buildEngine(avgIters, minIters, engine_path);
}

void* engine_create(const char* engine_path, cudaStream_t stream) {
    return new TRTEngine(engine_path, stream);
}

void  engine_destroy(void* engine) {
    delete static_cast<TRTEngine*>(engine);
}

size_t engine_batchsize(void* engine) {
    return static_cast<TRTEngine*>(engine)->batchsize();
}

BindingInfo engine_get_info(void* engine, size_t index) {
    return static_cast<TRTEngine*>(engine)->getBindingInfo(index);
}

void engine_add_ptr(void* engine, void* ptr) {
    static_cast<TRTEngine*>(engine)->addBindingPtr(ptr);
}

void engine_forward(void* engine) {
    static_cast<TRTEngine*>(engine)->forward();
}

size_t engine_n_bindings(void* engine) {
    return static_cast<TRTEngine*>(engine)->getNbBindings();
}