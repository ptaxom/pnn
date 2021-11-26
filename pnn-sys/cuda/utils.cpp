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

std::vector<std::string> load_classes(const char** c_classes) {
    std::vector<std::string> classes;
    size_t i = 0;
    while (c_classes[i]) classes.push_back(c_classes[i++]);
    return classes;
}

std::vector<BoundingBox> load_bboxes(size_t n_boxes, BoundingBox* const boxes) {
    std::vector<BoundingBox> bboxes;
    for (size_t i = 0; i < n_boxes; i++) 
        bboxes.push_back(boxes[i]);
    return bboxes;
}

void draw_bboxes(cv::Mat &image, const std::vector<BoundingBox> &bboxes, const std::vector<std::string> &classes) {
    cv::Size imsize = image.size();

    int thickness = std::max(1.0f, imsize.height * .002f);
    size_t n_boxes = bboxes.size();
    size_t n_classes = classes.size();

    for(size_t i = 0; i < n_boxes; i++) {
        cv::Point p1, p2, p_text;
        BoundingBox box = bboxes[i];

        p1.x = int(imsize.width * box.x0);
        p2.x = int(imsize.width * box.x1);
        p1.y = int(imsize.height * box.y0);
        p2.y = int(imsize.height * box.y1);

        int offset = box.class_id * 123457 % n_classes;
        float red = get_color(2, offset, n_classes);
        float green = get_color(1, offset, n_classes);
        float blue = get_color(0, offset, n_classes);
        float const font_size = imsize.height / 1000.F;
        cv::Size text_size = cv::getTextSize(classes[box.class_id], cv::FONT_HERSHEY_COMPLEX_SMALL, font_size, 1, 0);

        p_text.x = p1.x;
        p_text.y = p1.y - 4;
        cv::Scalar color(red * 256, green * 256, blue * 256);

        cv::rectangle(image, p1, p2, color, thickness, 8, 0);
        cv::putText(image, classes[box.class_id], p_text, cv::FONT_HERSHEY_COMPLEX_SMALL, font_size, color, 2 * font_size, cv::LINE_AA);
    }
}

int render_bboxes(const char* image_path, size_t n_boxes, void* const boxes, const char** classes, const char* window_name) {
    cv::Mat image = cv::imread(image_path);
    if (image.empty()) return 0;
    draw_bboxes(image,
        load_bboxes(n_boxes, (BoundingBox*)boxes),
        load_classes(classes)
    );
    cv::imshow(window_name, image);
    cv::waitKey();
    return 1;
}