#include "builder.hpp"

#include <mutex>
#include <map>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cuda.h>

#include "utils.hpp"

using Severity = ILogger::Severity;

std::map<Severity, std::string> SEVERITY_COLORS = {
    {Severity::kINTERNAL_ERROR, "\033[91m\033[1m[CRITICAL]: "},
    {Severity::kERROR,                 "\033[91m[ERROR]:    "},
    {Severity::kWARNING,               "\033[93m[WARNING]:  "},
    {Severity::kINFO,                  "\033[92m[INFO]:     "},
    {Severity::kVERBOSE,               "\033[94m[DEBUG]:    "}
};

std::ostream& operator<<(std::ostream& os, const nvinfer1::Dims& obj)
{
    for(int i = 0; i < obj.nbDims - 1; i++)
        os << obj.d[i] << "x";
    os << obj.d[obj.nbDims - 1];
    return os;
}

class EngineLogger : public ILogger {

    std::mutex log_guard;

public:

    Severity logger_severity = Severity::kWARNING;

    void log(Severity severity, const char* msg) noexcept override {
        std::lock_guard<std::mutex> guard(log_guard);
        
        if (severity <= logger_severity)
            std::cout << SEVERITY_COLORS[severity] << msg << "\033[0m" <<  std::endl;
    }

    template <class T>
    void log(Severity severity, const char* msg, T model_name){
        std::stringstream message_ss;
        message_ss << msg << " " << model_name;
        log(severity, message_ss.str().c_str());
    }

} EngineLogger;

void set_severity(int severity)
{
    if (severity < 0 || severity > 4)
        throw std::runtime_error("Unsupported severity");
    EngineLogger.logger_severity = (Severity)severity;
}


TRTBuilder::TRTBuilder(cudnnDataType_t dataType, int32_t maxBatchsize): mBatchSize(maxBatchsize) {
    switch (dataType)
    {
    case CUDNN_DATA_INT8:
        mDataType = DataType::kINT8;
        break;
    case CUDNN_DATA_HALF:
        mDataType = DataType::kHALF;
        break;
    case CUDNN_DATA_FLOAT:
        mDataType = DataType::kFLOAT;
        break;
    default:
        throw std::runtime_error("Unsupported dtype");
        break;
    }

    mBuilder = std::unique_ptr<IBuilder>(createInferBuilder(EngineLogger));
    if (!mBuilder) FatalError("Couldnt creater IBuilder");

    mBuilderConfig = std::unique_ptr<IBuilderConfig>(mBuilder->createBuilderConfig());
    if (!mBuilder) FatalError("Couldnt creater IBuilderConfig");

    mNetworkDefenition = std::unique_ptr<INetworkDefinition>(mBuilder->createNetworkV2(0));
    if (!mBuilder) FatalError("Couldnt creater INetworkDefinition");

    mBuilderConfig->setMaxWorkspaceSize(1U << 20);

    mBuilder->setMaxBatchSize(maxBatchsize);
    if (mDataType == DataType::kHALF) {
        mBuilderConfig->setFlag(BuilderFlag::kFP16);
    } else if (mDataType == DataType::kINT8) {
        mBuilderConfig->setFlag(BuilderFlag::kINT8);
    }
}

TRTBuilder::~TRTBuilder() {
}

int TRTBuilder::addLayer(ILayer* layer) {
    if (!layer) return -1;

    mLayers.push_back(layer);
    return mLayers.size() - 1;
}

int TRTBuilder::addConvolution(size_t input_id,  int feature_maps, int kernel_size, Weights kernel, Weights biases) {
    ITensor* tensor = mLayers[input_id]->getOutput(0);
    if (!tensor) return -1;

    ILayer* conv = mNetworkDefenition->addConvolutionNd(
        *tensor,
        feature_maps,
        Dims{2, {kernel_size, kernel_size}},
        kernel,
        biases
    );
    return addLayer(conv);
}

int TRTBuilder::addActivation(size_t input_id,  const std::string &activation_name) {
    ITensor* tensor = mLayers[input_id]->getOutput(0);
    if (!tensor) return -1;

    ILayer* layer = nullptr;
    
    if (activation_name == "mish") {
        IPluginCreator* creator = getPluginRegistry()->getPluginCreator("YOLOMishPlugin", "1");
        if (!creator) {
            std::cerr << "Couldnt find Mish Plugin for YOLO" << std::endl;
            return -1;
        }
        PluginFieldCollection *pluginData = nullptr; // Unused
        IPluginV2 *pluginObj = creator->createPlugin("mish", pluginData);

        layer = mNetworkDefenition->addPluginV2(&tensor, 1, *pluginObj);
    } else if (activation_name == "logistic") {
        layer = mNetworkDefenition->addActivation(*tensor, ActivationType::kSIGMOID);
    } else if (activation_name == "linear") { 
        return mLayers.size() - 1;
    }
    
    if (!tensor) return -1;

    return addLayer(layer);
}

int TRTBuilder::addShortcut(const std::vector<size_t> &input_ids) {
    if (input_ids.size() < 2) return -1;

    std::vector<ITensor*> inputs;
    for (const auto input_id: input_ids) {
        if (input_id >= mLayers.size()) return -1;

        auto tensor = mLayers[input_id]->getOutput(0);
        if (!tensor) return -1;

        inputs.push_back(tensor);
    }

    ILayer* shortcut = mNetworkDefenition->addElementWise(*inputs[0], *inputs[1], ElementWiseOperation::kSUM);
    if (!shortcut) return -1;
    mLayers.push_back(shortcut);

    for (size_t i = 2; i < input_ids.size(); i++) {
        ITensor* in1 = inputs[i];
        ITensor* in2 = mLayers[mLayers.size() - 1]->getOutput(0);

        ILayer* layer = mNetworkDefenition->addElementWise(*in1, *in2, ElementWiseOperation::kSUM);
        if (!layer) return -1;
        mLayers.push_back(layer);
    }

    return mLayers.size() - 1;
}

int TRTBuilder::addUpsample(size_t input_id,  size_t stride) {
    ITensor* tensor = mLayers[input_id]->getOutput(0);
    if (!tensor) return -1;

    ILayer *layer = mNetworkDefenition->addResize(*tensor);
    if (!layer) return -1;

    IResizeLayer* resize = dynamic_cast<IResizeLayer*>(layer);
    float scale = static_cast<float>(stride);
    float scales[4] = {1, 1, scale, scale};
    resize->setScales(scales, 4);
    resize->setResizeMode(ResizeMode::kNEAREST);

    return addLayer(resize);    
}

int TRTBuilder::addRoute(const std::vector<size_t> &input_ids) {
    if (input_ids.size() < 2) return -1;

    std::vector<ITensor*> inputs;
    for (const auto input_id: input_ids) {
        if (input_id >= mLayers.size()) return -1;

        auto tensor = mLayers[input_id]->getOutput(0);
        if (!tensor) return -1;

        inputs.push_back(tensor);
    }

    ILayer* layer = mNetworkDefenition->addConcatenation(inputs.data(), inputs.size());
    if (!layer) return -1;

    IConcatenationLayer* concat = dynamic_cast<IConcatenationLayer*>(layer);
    concat->setAxis(0); // Concat across chanell

    return addLayer(concat);
}

int TRTBuilder::addInput(const std::string &name, int32_t channels, int32_t height, int32_t width) {
    Dims dim{4, {mBatchSize, channels, height, width}};
    ITensor* input = mNetworkDefenition->addInput(name.c_str(), mDataType, dim);
    if (!input) return -1;

    ILayer* identety = mNetworkDefenition->addIdentity(*input);
    return addLayer(identety);
}


void TRTBuilder::addYolo(size_t input_id) {
    ITensor* tensor = mLayers[input_id]->getOutput(0);
    mNetworkDefenition->markOutput(*tensor);
}

int TRTBuilder::addPooling(size_t input_id, int32_t stride, int32_t window_size, int32_t padding, bool is_max) {
    ITensor* tensor = mLayers[input_id]->getOutput(0);
    if (!tensor) return -1;

    ILayer *layer = mNetworkDefenition->addPoolingNd(*tensor, is_max ? PoolingType::kMAX : PoolingType::kAVERAGE, Dims{2, {window_size, window_size}});
    if (!layer) return -1;

    IPoolingLayer* pool = dynamic_cast<IPoolingLayer*>(layer);
    pool->setStrideNd(Dims{4, {1, 1, stride, stride}});
    pool->setPaddingNd(Dims{4, {1, 1, padding, padding}});
    return addLayer(pool);
}

bool TRTBuilder::buildEngine(int32_t avgIters, int32_t minIters, const std::string &engine_path) {
    // Inputs and outputs should be in NCHW FP32 for compatibility
    for(int32_t i = 0; i < mNetworkDefenition->getNbInputs(); i++) {
        auto input = mNetworkDefenition->getInput(i);
        input->setType(DataType::kFLOAT);
        input->setAllowedFormats(1U << static_cast<int>(TensorFormat::kLINEAR));
    }
    for(int32_t i = 0; i < mNetworkDefenition->getNbOutputs(); i++) {
        auto output = mNetworkDefenition->getOutput(i);
        output->setType(DataType::kFLOAT);
        output->setAllowedFormats(1U << static_cast<int>(TensorFormat::kLINEAR));
    }

    mBuilderConfig->setMaxWorkspaceSize(1U << 20);
    mBuilderConfig->setFlag(BuilderFlag::kREFIT);
    mBuilderConfig->setProfilingVerbosity(ProfilingVerbosity::kLAYER_NAMES_ONLY);
    mBuilderConfig->setAvgTimingIterations(avgIters);
    mBuilderConfig->setMinTimingIterations(minIters);
    
    IHostMemory* serialized = mBuilder->buildSerializedNetwork(*mNetworkDefenition, *mBuilderConfig);

    if (!serialized) return false;

    std::ofstream engineFile(engine_path, std::ios::binary);
    if (!engineFile) {
        std::cerr << "Couldnt open " << engine_path << " for serialization" << std::endl;
    }
    engineFile.write(static_cast<char*>(serialized->data()), serialized->size());
    return !engineFile.fail();
}