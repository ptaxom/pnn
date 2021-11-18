#include "kernels.h"
#include "math.h"

// #TODO: Add more accurate estimation
dim3 get_gridsize(size_t elements){
    unsigned int required_blocks = (elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if(required_blocks <= 65535){
        return {required_blocks, 1, 1};
    }
    unsigned int proposed_width = ceil(sqrt(required_blocks));
    unsigned int required_height = (required_blocks - proposed_width + 1) / proposed_width;
    return {proposed_width, required_height, 1};
}

#include <opencv2/opencv.hpp>

// WORK ONLY WITH FLOAT!!!
int load_image2batch(const char* image_path, size_t batch_id, int width, int height, void* input_data) {
    cv::Mat image = cv::imread(image_path);
    if (image.empty()) return 0;

    cv::resize(image, image, cv::Size(width, height));
    if (image.empty()) return 0;

    image.convertTo(image, CV_32FC3, 1 / 255.0);
    if (image.empty()) return 0;

    cv::Mat channels[3];
    cv::split(image, channels);
    size_t channel_size = width * height;
    size_t sample_size = 3 * channel_size;
    for(int channel = 0; channel < 3; channel++) {

        cudaError_t status = cudaMemcpy((float*)input_data + batch_id * sample_size + channel * channel_size, 
            (void*)channels[2 - channel].data,
            channel_size * sizeof(float),
            cudaMemcpyHostToDevice
        );
        if (status != cudaSuccess) 
            return 0;
    }
    return 1;
}