#pragma once

#include "trt.h"
#include "mish.h"

#include <string>
#include <vector>
#include <NvInferRuntimeCommon.h>

using namespace nvinfer1;

typedef enum CustomActivationType_t {
    MISH = 1,
} CustomActivationType;

class ActivationPlugin : public IPluginV2
{
public:
    ActivationPlugin(const std::string name, CustomActivationType type);

    ActivationPlugin(const std::string name, const void* data, size_t length);

    // It doesn't make sense to make ActivationPlugin without arguments, so we delete default constructor.
    ActivationPlugin() = delete;

    int getNbOutputs() const noexcept override;

    Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) noexcept override;

    int initialize() noexcept override;

    void terminate() noexcept override;

    size_t getWorkspaceSize(int) const noexcept override
    {
        return 0;
    };

    int enqueue(int batchSize, const void* const* inputs, void* const* outputs, void* workspace,
        cudaStream_t stream) noexcept override;

    size_t getSerializationSize() const noexcept override;

    void serialize(void* buffer) const noexcept override;

    void configureWithFormat(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs, DataType type,
        PluginFormat format, int maxBatchSize) noexcept override;

    bool supportsFormat(DataType type, PluginFormat format) const noexcept override;

    const char* getPluginType() const noexcept override;

    const char* getPluginVersion() const noexcept override;

    void destroy() noexcept override;

    nvinfer1::IPluginV2* clone() const noexcept override;

    void setPluginNamespace(const char* pluginNamespace) noexcept override;

    const char* getPluginNamespace() const noexcept override;

private:
    const std::string mLayerName;
    CustomActivationType mActivation;
    DataType mOpType;
    size_t mInputVolume;
    std::string mNamespace;
    cudaError_t (*inference_call)(cudaStream_t, size_t, const void*, void*);
};

class ActivationCreator: public IPluginCreator {
public:

    ActivationCreator();

    const char* getPluginName() const noexcept override;

    const char* getPluginVersion() const noexcept override;

    const PluginFieldCollection* getFieldNames() noexcept override;

    IPluginV2* createPlugin(const char* name, const PluginFieldCollection* fc) noexcept override;

    IPluginV2* deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept override;

    void setPluginNamespace(const char* pluginNamespace) noexcept override;

    const char* getPluginNamespace() const noexcept override;

private:
    std::string mNamespace;
    static PluginFieldCollection mFC;
    static std::vector<PluginField> mPluginAttributes;
};