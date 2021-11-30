#include "trt.h"
#include "builder.hpp"
#include "engine.hpp"
//dummy
int main() {
    // TRTBuilder builder(CUDNN_DATA_HALF, 4);
    // int idx = builder.addInput("input_0", 3, 32, 32);
    // idx = builder.addActivation(idx, "mish");
    // builder.addYolo(idx);
    // bool p = builder.buildEngine(8, 1, "test.engine");

    TRTEngine engine("test.engine");
    
    // printf("%ld\n", p);
}