#include "engine.hpp"

#include <fstream>

#include "utils.hpp"

using namespace nvinfer1;

TRTEngine::TRTEngine(const std::string &engineFileName, cudaStream_t stream): mStream(stream) {

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

    mBatchsize = mEngine->getMaxBatchSize();
}
    
TRTEngine::~TRTEngine() {
}

size_t TRTEngine::batchsize() {
    return mBatchsize;
}

size_t TRTEngine::getNbBindings() {
    return mEngine->getNbBindings();
}

BindingInfo TRTEngine::getBindingInfo(const size_t index) {
    Dims dim = mEngine->getBindingDimensions(index);
    if (dim.nbDims != 4)
        throw std::runtime_error("Supported only engines with shape 4");

    BindingInfo info{
        mEngine->getBindingName(index),
        batchsize(),
        static_cast<size_t>(dim.d[1]),
        static_cast<size_t>(dim.d[2]),
        static_cast<size_t>(dim.d[3]),
        mEngine->bindingIsInput(index)
    };
    return info;
}

void TRTEngine::addBindingPtr(void* ptr) {
    mBindings.push_back(ptr);
}

void TRTEngine::forward() {
    mContext->enqueueV2(mBindings.data(), mStream, nullptr);
}