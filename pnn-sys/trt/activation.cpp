#include "activation.hpp"
#include "utils.hpp"

#include <cassert>
#include <cstring>
#include <iostream>

using namespace nvinfer1;

namespace
{
const char* ACTIVATION_PLUGIN_VERSION{"1"};
const char* ACTIVATION_PLUGIN_NAME{"YOLOMishPlugin"};
}

PluginFieldCollection ActivationCreator::mFC{};
std::vector<PluginField> ActivationCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(ActivationCreator);


ActivationPlugin::ActivationPlugin(const std::string name, CustomActivationType type):
    mLayerName(name),
    mActivation(type),
    inference_call(&activation_mish_fp32) {

}

ActivationPlugin::ActivationPlugin(const std::string name, const void* data, size_t length)
    : mLayerName(name)
{
    const char* d = static_cast<const char*>(data);
    const char* a = d;

    mActivation = static_cast<CustomActivationType>(readFromBuffer<uint8_t>(d));

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
    return 1 * sizeof(uint8_t);
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
    status = inference_call(stream, mInputVolume * batchSize, inputs[0], output);

    return status;
}

void ActivationPlugin::terminate() noexcept {}

void ActivationPlugin::destroy() noexcept
{
    delete this;
}

IPluginV2* ActivationPlugin::clone() const noexcept
{
    auto plugin = new ActivationPlugin(mLayerName, mActivation);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

void ActivationPlugin::setPluginNamespace(const char* libNamespace) noexcept
{
    mNamespace = libNamespace;
}

const char* ActivationPlugin::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

bool ActivationPlugin::supportsFormat(DataType type, PluginFormat format) const noexcept
{
    // This plugin only supports ordinary floats, and NCHW input format
    if ((type == DataType::kFLOAT || type == DataType::kHALF) && format == PluginFormat::kLINEAR)
        return true;
    else
        return false;
}

void ActivationPlugin::serialize(void* buffer) const noexcept
{
    char* d = static_cast<char*>(buffer);
    const char* a = d;

    uint8_t act = static_cast<uint8_t>(mActivation);

    writeToBuffer(d, act);

    assert(d == a + getSerializationSize());
}

void ActivationPlugin::configureWithFormat(const Dims* inputs, int nbInputs, const Dims* outputs, int nbOutputs,
    DataType type, PluginFormat format, int) noexcept
{
    // Validate input arguments
    assert(nbOutputs == 1);
    assert(type == DataType::kFLOAT || type == DataType::kHALF);
    assert(format == PluginFormat::kLINEAR);

    // Fetch volume for future enqueue() operations
    size_t volume = 1;
    for (int i = 0; i < inputs->nbDims; i++)
    {
        volume *= inputs->d[i];
    }
    mInputVolume = volume;
    if (type == DataType::kHALF) {
        inference_call = &activation_mish_fp16;
        std::cout << "Working in FP16 mode(DEBUG)" << std::endl;
    }
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