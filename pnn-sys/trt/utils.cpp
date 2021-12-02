#include "utils.hpp"

#include <mutex>
#include <map>
#include <fstream>
#include <sstream>

#include <NvInferPlugin.h>

#include "activation.hpp"

using namespace nvinfer1;

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

std::unique_ptr<nvinfer1::IBuilder> getIBuilder() {
    return std::unique_ptr<nvinfer1::IBuilder>(createInferBuilder(EngineLogger));
}

std::unique_ptr<nvinfer1::IRuntime> getIRuntime() {
    return std::unique_ptr<nvinfer1::IRuntime>(createInferRuntime(EngineLogger));
}

ActivationCreator ACTIVATION_CREATOR; 
bool ACTIVATION_REGISTERED = false;

void init_plugins() {
    initLibNvInferPlugins((void*)&EngineLogger, "");
    if (!ACTIVATION_REGISTERED) {
        bool registered = getPluginRegistry()->registerCreator(ACTIVATION_CREATOR, "");
        if (!registered) {
            std::cerr << "Couldnt register ActivationPlugin" << std::endl;
        } else {
            std::cout << "Registered plugin" << std::endl;
        }
        ACTIVATION_REGISTERED = true;
    }
}