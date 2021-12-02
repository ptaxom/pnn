#include <NvInfer.h>

#include <cudnn.h>
#include <memory>
#include <map>

#include "activation.hpp"

using namespace nvinfer1;

class TRTBuilder {
public:
    TRTBuilder(cudnnDataType_t dataType, int32_t maxBatchsize);

    ~TRTBuilder();

    int addConvolution(size_t input_id, int feature_maps, int kernel_size, int padding, int stride, Weights kernel, Weights biases);

    int addActivation(size_t input_id,  const std::string &activation_name);

    int addShortcut(const std::vector<size_t> &input_ids);

    int addUpsample(size_t input_id,  size_t stride);

    int addInput(const std::string &name, int32_t channels, int32_t height, int32_t width);

    void addYolo(size_t input_id, const std::string &name);

    int addRoute(const std::vector<size_t> &input_ids);

    int addPooling(size_t input_id, int32_t stride, int32_t window_size, int32_t padding, bool is_max);

    bool buildEngine(int32_t avgIters, int32_t minIters, const std::string &engine_path);

private:
    void setLayerName(ILayer* layer, const std::string &prefix);

    int addLayer(ILayer* layer);

private:
    DataType mDataType;
    int32_t mBatchSize;

    std::unique_ptr<INetworkDefinition> mNetworkDefenition;
    std::unique_ptr<IBuilder> mBuilder;
    std::unique_ptr<IBuilderConfig> mBuilderConfig;
    std::unique_ptr<IRuntime> mRuntime;

    // I think this pointers are owner by mNetworkDefenetion, but need double-check
    std::vector<ILayer*> mLayers;
    std::map<std::string, size_t> mCounter;

};