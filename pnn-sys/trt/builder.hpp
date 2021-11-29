#include <NvInfer.h>

#include <cudnn.h>
#include <memory>

#include "activation.hpp"

using namespace nvinfer1;

class TRTBuilder {
public:
    TRTBuilder(cudnnDataType_t dataType);

    ~TRTBuilder();

    int addConvolution(size_t input_id,  int feature_maps, int kernel_size, Weights kernel, Weights biases);

    int addActivation(size_t input_id,  const std::string &activation_name);

    int addShortcut(const std::vector<size_t> &input_ids);

    int addUpsample(size_t input_id,  size_t stride);

    int addInput(const std::string &name, int32_t channels, int32_t height, int32_t width);

    void addYolo(size_t input_id);

    int addRoute(const std::vector<size_t> &input_ids);

private:
    int addLayer(ILayer* layer);

private:
    DataType mDataType;

    std::unique_ptr<INetworkDefinition> mNetworkDefenition;
    std::unique_ptr<IBuilder> mBuilder;
    std::unique_ptr<IBuilderConfig> mBuilderConfig;
    std::unique_ptr<IRuntime> mRuntime;

    // I think this pointers are owner by mNetworkDefenetion, but need double-check
    std::vector<ILayer*> mLayers;

};