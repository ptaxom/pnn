#include "engine.hpp"

#include <fstream>

#include "utils.hpp"

TRTEngine::TRTEngine(const std::string &engineFileName) {
    checkCuda(cudaStreamCreate(&mStream));

    mRuntime = getIRuntime();
    if (!mRuntime) FatalError("Couldnt acquire IRuntime");

    std::ifstream engineFile(engineFileName, std::ios::binary);
    if (!engineFile) FatalError("Couldnt open engine file");

    engineFile.seekg(0, engineFile.end);
    size_t size = engineFile.tellg();
    engineFile.seekg(0, engineFile.beg);

    char* engine = new char[size];
    engineFile.read(engine, size);
    engineFile.close();


    mEngine = std::unique_ptr<nvinfer1::ICudaEngine>(mRuntime->deserializeCudaEngine(static_cast<const void*>(engine), size));
    mContext = std::unique_ptr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
    delete[] engine;
}
    
TRTEngine::~TRTEngine() {
    cudaStreamDestroy(mStream);
}