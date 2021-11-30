#pragma once

#include <iostream>
#include <sstream>
#include <memory>

#include <NvInfer.h>


#define FatalError(s) {                                                \
    std::stringstream _where, _message;                                \
    _where << __FILE__ << ':' << __LINE__;                             \
    _message << std::string(s) + "\n" << __FILE__ << ':' << __LINE__;  \
    cudaDeviceReset();                                                 \
    throw std::runtime_error(_message.str());                          \
}

#define checkCuda(status) {                                            \
    std::stringstream _error;                                          \
    if (status != 0) {                                                 \
      _error << "Cuda failure: "<<cudaGetErrorString(status);          \
      FatalError(_error.str());                                        \
    }                                                                  \
}

void set_severity(int severity);

std::unique_ptr<nvinfer1::IBuilder> getIBuilder();

std::unique_ptr<nvinfer1::IRuntime> getIRuntime();