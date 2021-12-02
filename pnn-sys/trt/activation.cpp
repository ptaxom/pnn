#include "activation.hpp"
#include "utils.hpp"

#include <cassert>
#include <cstring>
#include <iostream>

using namespace nvinfer1;

template <typename T>
void writeToBuffer(char*& buffer, const T& val)
{
    *reinterpret_cast<T*>(buffer) = val;
    buffer += sizeof(T);
}

template <typename T>
T readFromBuffer(const char*& buffer)
{
    T val = *reinterpret_cast<const T*>(buffer);
    buffer += sizeof(T);
    return val;
}



namespace {
const char* ACTIVATION_PLUGIN_VERSION{"1"};
const char* ACTIVATION_PLUGIN_NAME{"YOLOMishPlugin"};
}

PluginFieldCollection ActivationCreator::mFC{};
std::vector<PluginField> ActivationCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(ActivationCreator);


ActivationPlugin::ActivationPlugin(const std::string name, CustomActivationType type):
    mLayerName(name),
    mActivation(type) {
    mOpType = DataType::kFLOAT;
    mInputVolume = 0;
}

ActivationPlugin::ActivationPlugin(const std::string name, const void* data, size_t length)
    : mLayerName(name)
{
    const char* d = static_cast<const char*>(data);
    const char* a = d;

    mActivation  = static_cast<CustomActivationType>(readFromBuffer<int32_t>(d));
    mOpType      = static_cast<DataType>(readFromBuffer<int32_t>(d));
    mInputVolume = readFromBuffer<size_t>(d); // TODO: Auto infer??

    assert(d == (a + length));
}

const char* ActivationPlugin::getPluginType() const noexcept
{
    return ACTIVATION_PLUGIN_NAME;
}

const char* ActivationPlugin::getPluginVersion() const noexcept
{
    return ACTIVATION_PLUGIN_VERSION;
}

int ActivationPlugin::getNbOutputs() const noexcept
{
    return 1;
}

size_t ActivationPlugin::getSerializationSize() const noexcept
{
    return  2 * sizeof(uint32_t) + sizeof(size_t);
}

Dims ActivationPlugin::getOutputDimensions(int index, const Dims* inputs, int nbInputDims) noexcept
{
    assert(nbInputDims == 1);
    assert(index == 0);

    return *inputs;
}

int ActivationPlugin::initialize() noexcept
{
    return 0;
}

int ActivationPlugin::enqueue(int batchSize, const void* const* inputs, void* const* outputs, void*, cudaStream_t stream) noexcept
{
    int status = -1;

    // Our plugin outputs only one tensor
    void* output = outputs[0];

    // Launch CUDA kernel wrapper and save its return value
    if (mOpType == DataType::kHALF) {
        status = trt_activation_mish_fp16(stream, mInputVolume * batchSize, inputs[0], output);
    } else {
        status = trt_activation_mish_fp32(stream, mInputVolume * batchSize, inputs[0], output);
    }

    return status;
}

void ActivationPlugin::terminate() noexcept {}

void ActivationPlugin::destroy() noexcept
{
    delete this;
}

IPluginV2Ext* ActivationPlugin::clone() const noexcept
{
    return new ActivationPlugin(*this);
}

void ActivationPlugin::setPluginNamespace(const char* libNamespace) noexcept
{
    mNamespace = libNamespace;
}

const char* ActivationPlugin::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

bool ActivationPlugin::supportsFormat(DataType type, PluginFormat format) const noexcept {
    if ((type == DataType::kFLOAT || type == DataType::kHALF) && format == PluginFormat::kLINEAR)
        return true;
    else
        return false;
}

void ActivationPlugin::serialize(void* buffer) const noexcept
{
    char* d = static_cast<char*>(buffer);
    const char* a = d;

    int32_t act   = static_cast<int32_t>(mActivation);
    int32_t dtype = static_cast<int32_t>(mOpType); 

    writeToBuffer(d, act);
    writeToBuffer(d, dtype);
    writeToBuffer(d, mInputVolume);

    assert(d == a + getSerializationSize());
}

void ActivationPlugin::configureWithFormat(const Dims* inputs, int nbInputs, const Dims* outputs, int nbOutputs,
    DataType type, PluginFormat format, int) noexcept
{
    // Validate input arguments
    assert(nbOutputs == 1);
    assert(type == DataType::kFLOAT || type == DataType::kHALF);
    assert(format == PluginFormat::kLINEAR);

    mOpType = type == DataType::kHALF ? DataType::kHALF : DataType::kFLOAT;
    mInputVolume = dim2size(inputs[0]);
}

nvinfer1::DataType ActivationPlugin::getOutputDataType(int32_t index, nvinfer1::DataType const *inputTypes, int32_t nbInputs) const noexcept {
    auto dtype = inputTypes[0];
    if (dtype != DataType::kHALF)
        dtype = DataType::kFLOAT;
    return dtype;
}

bool ActivationPlugin::isOutputBroadcastAcrossBatch (int32_t outputIndex, bool const *inputIsBroadcasted, int32_t nbInputs) const noexcept {
    bool is_broadcasted = inputIsBroadcasted[0];
    for (int i = 1; i < nbInputs && is_broadcasted; i++)
        is_broadcasted &= inputIsBroadcasted[i];
    return is_broadcasted;
}

bool ActivationPlugin::canBroadcastInputAcrossBatch (int32_t inputIndex) const noexcept {
    return true;
}

void ActivationPlugin::configurePlugin (Dims const *inputDims, 
    int32_t nbInputs, 
    Dims const *outputDims, 
    int32_t nbOutputs, 
    DataType const *inputTypes, 
    DataType const *outputTypes, 
    bool const *inputIsBroadcast, 
    bool const *outputIsBroadcast,
    PluginFormat floatFormat, 
    int32_t maxBatchSize
) noexcept {
    assert(nbInputs == 1);
    assert(nbOutputs == 1);

    auto dtype = inputTypes[0];
    mOpType = dtype == DataType::kHALF ? DataType::kHALF : DataType::kFLOAT;
    mInputVolume = dim2size(inputDims[0]);
}

ActivationCreator::ActivationCreator() {
    mPluginAttributes.emplace_back(PluginField("activation", nullptr, PluginFieldType::kINT8, 1));

    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* ActivationCreator::getPluginName() const noexcept
{
    return ACTIVATION_PLUGIN_NAME;
}

const char* ActivationCreator::getPluginVersion() const noexcept
{
    return ACTIVATION_PLUGIN_VERSION;
}


void ActivationCreator::setPluginNamespace(const char* libNamespace) noexcept
{
    mNamespace = libNamespace;
}

const char* ActivationCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

const PluginFieldCollection* ActivationCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2* ActivationCreator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept
{
    if (strcmp(name, "mish") == 0) {
        //TODO: Is it necessary??
        return new ActivationPlugin(name, MISH);
    }
    return nullptr;
}

IPluginV2* ActivationCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept
{
    return new ActivationPlugin(name, serialData, serialLength);
}