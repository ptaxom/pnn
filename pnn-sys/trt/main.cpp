#include "trt.h"
#include "builder.hpp"
#include "engine.hpp"
//dummy
int main() {
    TRTBuilder builder(CUDNN_DATA_HALF, 4);
    int idx = builder.addInput("input_0", 3, 32, 32);
    idx = builder.addActivation(idx, "mish");
    builder.addYolo(idx, "d");
    bool p = builder.buildEngine(8, 1, "test.engine");
    printf("%ld\n", p);

    // TRTEngine engine("test.engine", 0);
    
}