#pragma once

#include <memory>
#include <NvInfer.h>

#include <cuda.h>

class TRTEngine {
public:
    TRTEngine(const std::string &engineFile);
    
    ~TRTEngine();

private:
    std::unique_ptr<nvinfer1::IRuntime> mRuntime;
    std::unique_ptr<nvinfer1::ICudaEngine> mEngine;
    std::unique_ptr<nvinfer1::IExecutionContext> mContext;
    
    cudaStream_t mStream;

};