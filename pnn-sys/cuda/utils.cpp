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
struct BoundingBox{
    float x0;
    float y0;
    float x1;
    float y1;
    size_t class_id;
    float probability;
    float objectness;
};

const float colors[6][3] = { {1,0,1}, {0,0,1},{0,1,1},{0,1,0},{1,1,0},{1,0,0} };

float get_color(int c, int x, int max)
{
    float ratio = ((float)x/max)*5;
    int i = floor(ratio);
    int j = ceil(ratio);
    ratio -= i;
    float r = (1-ratio) * colors[i][c] + ratio*colors[j][c];
    return r;
}

int render_bboxes(const char* image_path, size_t n_boxes, void* const boxes, const char** classes, const char* window_name) {
    cv::Mat image = cv::imread(image_path);
    if (image.empty()) return 0;
    cv::Size imsize = image.size();

    int thickness = std::max(1.0f, imsize.height * .002f);

    for(int i = 0; i < n_boxes; i++) {
        cv::Point p1, p2, p_text;
        BoundingBox box = ((BoundingBox*)boxes)[i];

        p1.x = int(imsize.width * box.x0);
        p2.x = int(imsize.width * box.x1);
        p1.y = int(imsize.height * box.y0);
        p2.y = int(imsize.height * box.y1);

        int offset = box.class_id * 123457 % n_boxes;
        float red = get_color(2, offset, n_boxes);
        float green = get_color(1, offset, n_boxes);
        float blue = get_color(0, offset, n_boxes);
        float const font_size = imsize.height / 1000.F;
        cv::Size text_size = cv::getTextSize(classes[box.class_id], cv::FONT_HERSHEY_COMPLEX_SMALL, font_size, 1, 0);

        p_text.x = p1.x;
        p_text.y = p1.y - 4;
        cv::Scalar color(red * 256, green * 256, blue * 256);

        cv::rectangle(image, p1, p2, color, thickness, 8, 0);
        cv::putText(image, classes[box.class_id], p_text, cv::FONT_HERSHEY_COMPLEX_SMALL, font_size, color, 2 * font_size, cv::LINE_AA);
    }
    cv::imshow(window_name, image);
    cv::waitKey();
    return 1;
}