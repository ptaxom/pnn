#include "builder.hpp"

#include <iostream>
#include <fstream>
#include <cuda.h>

#include "utils.hpp"


TRTBuilder::TRTBuilder(cudnnDataType_t dataType, int32_t maxBatchsize): mBatchSize(maxBatchsize) {
    init_plugins();
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

    mBuilder = std::unique_ptr<IBuilder>(getIBuilder());
    if (!mBuilder) FatalError("Couldnt creater IBuilder");

    mBuilderConfig = std::unique_ptr<IBuilderConfig>(mBuilder->createBuilderConfig());
    if (!mBuilder) FatalError("Couldnt creater IBuilderConfig");

    auto flag = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);

    mNetworkDefenition = std::unique_ptr<INetworkDefinition>(mBuilder->createNetworkV2(flag));
    if (!mBuilder) FatalError("Couldnt creater INetworkDefinition");

    mBuilder->setMaxBatchSize(maxBatchsize);
    if (mDataType == DataType::kHALF) {
        mBuilderConfig->setFlag(BuilderFlag::kFP16);
    } else if (mDataType == DataType::kINT8) {
        mBuilderConfig->setFlag(BuilderFlag::kINT8);
    }
}

TRTBuilder::~TRTBuilder() {
}

void TRTBuilder::setLayerName(ILayer* layer, const std::string &prefix) {
    std::stringstream ss;
    if (mCounter.find(prefix) == mCounter.end()) mCounter[prefix] = 0;
    ss << prefix << "_" << mCounter[prefix];
    ++mCounter[prefix];
    layer->setName(ss.str().c_str());
}

int TRTBuilder::addLayer(ILayer* layer) {
    if (!layer) return -1;

    mLayers.push_back(layer);
    return mLayers.size() - 1;
}

int TRTBuilder::addConvolution(size_t input_id,  int feature_maps, int kernel_size, int padding, int stride, Weights kernel, Weights biases) {
    ITensor* tensor = mLayers[input_id]->getOutput(0);
    if (!tensor) return -1;

    ILayer* layer = mNetworkDefenition->addConvolutionNd(
        *tensor,
        feature_maps,
        Dims{2, {kernel_size, kernel_size}},
        kernel,
        biases
    );
    if (!layer) return -1;
    IConvolutionLayer* conv = dynamic_cast<IConvolutionLayer*>(layer);
    conv->setPaddingNd(Dims{2, {padding, padding}});
    conv->setStrideNd(Dims{2, {stride, stride}});
    setLayerName(layer, "Convolution");

    return addLayer(layer);
}

int TRTBuilder::addActivation(size_t input_id,  const std::string &activation_name) {
    ITensor* tensor = mLayers[input_id]->getOutput(0);
    if (!tensor) return -1;

    ILayer* layer = nullptr;
    
    if (activation_name == "mish") {
        IPluginCreator* creator = getPluginRegistry()->getPluginCreator("YOLOMishPlugin", "1", "");
        if (!creator) {
            std::cerr << "Couldnt find Mish Plugin for YOLO" << std::endl;
            return -1;
        }
        PluginFieldCollection *pluginData = nullptr; // Unused
        IPluginV2 *pluginObj = creator->createPlugin("mish", pluginData);

        layer = mNetworkDefenition->addPluginV2(&tensor, 1, *pluginObj);
        if (!layer) return -1;
        setLayerName(layer, "Mish");
    } else if (activation_name == "logistic") {
        layer = mNetworkDefenition->addActivation(*tensor, ActivationType::kSIGMOID);
        if (!layer) return -1;
        setLayerName(layer, "Logistic");
    } else if (activation_name == "linear") { 
        return mLayers.size() - 1;
    }

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
    setLayerName(layer, "Upsample");

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
    setLayerName(layer, "Route");

    IConcatenationLayer* concat = dynamic_cast<IConcatenationLayer*>(layer);
    concat->setAxis(1); // Concat across chanell WTF????? https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/classnvinfer1_1_1_i_concatenation_layer.html#a27152d1ea5cf0ee387acd6adafe68f2d

    return addLayer(concat);
}

int TRTBuilder::addInput(const std::string &name, int32_t channels, int32_t height, int32_t width) {
    Dims dim{4, {mBatchSize, channels, height, width}};
    ITensor* input = mNetworkDefenition->addInput(name.c_str(), DataType::kFLOAT, dim);
    if (!input) return -1;

    ILayer* identety = mNetworkDefenition->addIdentity(*input);
    setLayerName(identety, "Identety");
    return addLayer(identety);
}


void TRTBuilder::addYolo(size_t input_id, const std::string &name) {
    ITensor* tensor = mLayers[input_id]->getOutput(0);
    tensor->setType(DataType::kFLOAT);
    tensor->setName(name.c_str());
    mNetworkDefenition->markOutput(*tensor);
}

int TRTBuilder::addPooling(size_t input_id, int32_t stride, int32_t window_size, int32_t padding, bool is_max) {
    ITensor* tensor = mLayers[input_id]->getOutput(0);
    if (!tensor) return -1;

    ILayer *layer = mNetworkDefenition->addPoolingNd(*tensor, is_max ? PoolingType::kMAX : PoolingType::kAVERAGE, Dims{2, {window_size, window_size}});
    if (!layer) return -1;
    setLayerName(layer, "Pooling");

    IPoolingLayer* pool = dynamic_cast<IPoolingLayer*>(layer);
    pool->setStrideNd(Dims{2, {stride, stride}});
    pool->setPaddingNd(Dims{2, {padding, padding}});
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

    mBuilderConfig->setMaxWorkspaceSize(1U << 30);
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