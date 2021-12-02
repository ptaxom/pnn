#pragma once

#include <memory>
#include <vector>

#include <cuda.h>
#include <NvInfer.h>

#include "structs.h"

class TRTEngine {
public:
    TRTEngine(const std::string &engineFile, cudaStream_t stream);
    
    ~TRTEngine();

    size_t batchsize();

    BindingInfo getBindingInfo(const size_t index);

    void addBindingPtr(void* ptr);

    void forward();

    size_t getNbBindings();

private:
    std::unique_ptr<nvinfer1::IRuntime> mRuntime;
    std::unique_ptr<nvinfer1::ICudaEngine> mEngine;
    std::unique_ptr<nvinfer1::IExecutionContext> mContext;
    
    cudaStream_t mStream;
    std::vector<void*> mBindings;

    int32_t mBatchsize;

};