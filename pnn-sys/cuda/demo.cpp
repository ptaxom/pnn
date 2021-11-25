#include "kernels.h"
#include <opencv2/opencv.hpp>
#include <queue>
#include <chrono>
#include <thread>
#include <condition_variable>

class DataLoader {
public:
    virtual cv::Mat* load() = 0;
};

class VideoLoader: public DataLoader {
public:
    virtual cv::Mat* load() {
        cv::Mat* frame = new cv::Mat;
        if (!capture_.read(*frame)) {
            return nullptr;
        }
        return frame;
    }

    VideoLoader(const std::string &video_path) : DataLoader() {
        capture_ = cv::VideoCapture(video_path);
        if (!capture_.isOpened()) 
            throw std::runtime_error("Couldnt open video");
    }

    ~VideoLoader() {
        capture_.release();
    }

private:
    cv::VideoCapture capture_;
};

class DataProcesser {
public:
    virtual bool postprocess(cv::Mat *image, std::vector<BoundingBox> boxes) = 0;
};

class DetectionRenderer : public DataProcesser {
public:
    virtual bool postprocess(cv::Mat *image, std::vector<BoundingBox> boxes) {
        
    }

    DetectionRenderer(const std::vector<std::string> &class_names): DataProcesser(), class_names_(class_names) {}

private:
    std::vector<std::string> class_names_;
};

template<typename T>
class Queue{
public:
    Queue(const size_t max_size=0): max_size_(max_size), poisoned_(false) {}

    void push(const &T item) {
        std::unique_lock<std::mutex> lk(mutex_);
        if (max_size_ > 0 && queue_.size() >= max_size_) {
            cv_.wait(lk, []{queue_.size() < max_size});
        }
        queue_.push(item);
        lk.unlock();
        cv_.notify_one();
    }

    const T& pop() {
        std::unique_lock<std::mutex> lk(mutex_);
        if (queue_.size() == 0)
            cv_.wait(lk, []{queue_.size() > 0});
        auto item = queue_.front();
        queue_.pop();
        lk.unlock();
        cv_.notify_one();
        return item;
    }

    void poison() {
        mutex_.lock();
        poisoned_ = true;
        mutex_.unlock();
    }

    bool is_poisoned() {
        std::lock_guard<std::mutex> lock(mutex_);
        return poisoned_;
    }

private:
    size_t max_size_;
    bool poisoned_;
    std::queue<T> queue_;
    std::mutex mutex_;
    std::condition_variable cv_;
};


const size_t MAX_LOAD_QUEUE_SIZE = 10; // Bufferization up to 10 frames per sample in batch

class ThreadedProcesser {
public:

    ThreadedProcesser(DataLoader *loader, 
                      DataProcesser *processer,
                      size_t batchsize,
                      size_t width,
                      size_t height,
                      void* inp_ptr,
                      void* model_ptr,
                      BoundingBox* (*inference_call)(void* model, size_t *n_boxes)
                      ): 
        loader_(loader), processer_(processer), batchsize_(batchsize),
        width_(width), height_(height), inp_ptr_(inp_ptr),
        downloaded_(false), model_ptr_(model_ptr),
        inference_call_(inference_call),
    {
        n_boxes_ = new size_t[batchsize];
        loaded_queue_ = Queue(MAX_LOAD_QUEUE_SIZE * batchsize_);
        
    }

    void start() {
        // std::thread loader = 
    }

    ~ThreadedProcesser() {
        delete n_boxes_[];
    }

private:
    

    static void freeze_thread(size_t ms) {
        if (ms <= 0)
            return;
        std::chrono::milliseconds timespan(ms);
        std::this_thread::sleep_for(timespan);
    }
    
    void load_thread() {
        while (1) {
            auto frame = loader_->load();

            load_mutex_.lock();
            if (loaded_queue_.size() < MAX_LOAD_QUEUE_SIZE * batchsize_) {
                loaded_queue_.push(frame);
                load_mutex_.unlock();
                if (!frame)
                    break; // Poisoning
            } else {
                load_mutex_.unlock();
                freeze_thread(LOAD_RETRY_MS);
            }
        }
    }

    void preprocess_thread() {
        std::queue<std::pair<cv::Mat, cv::Mat*>> frames;
        bool poisoned = false;
        while (1) {
            // If collected not enough preprocessed frames
            if (frames.size() < batchsize_ && !poisoned) {
                load_mutex_.lock();
                if (loaded_queue_.size() > 0) {
                    auto frame = loaded_queue_.front(); 
                    loaded_queue_.pop();
                    load_mutex_.unlock();
                    
                    if (!frame) {
                        poisoned = true;
                        continue;
                    }
                        

                    cv::Mat preprocessed_image;
                    cv::resize(preprocessed_image, *frame, cv::Size(width_, height_));
                    if (preprocessed_image.empty()) exit(101);

                    preprocessed_image.convertTo(preprocessed_image, CV_32FC3, 1 / 255.0);
                    if (preprocessed_image.empty()) exit(101);

                    cv::Mat channels[3];
                    cv::split(preprocessed_image, channels);
                    std::swap(channels[0], channels[2]);
                    cv::merge(channels, 3, preprocessed_image);
                    frames.push(std::make_pair(preprocessed_image, frame));
                }
            } else {
                inference_mutex_.lock();
                if (downloaded_) {
                    inference_mutex_.unlock();
                    freeze_thread(DOWNLOAD_RETRY_MS);
                } else {
                    size_t offset = 0;
                    size_t sample_size = 3 * width_ * height_ * sizeof(float);

                    while (frames.size() > 0) {
                        auto pair = frames.front();
                        frames.pop();
                        cudaMemcpy(inp_ptr_ + offset, pair.first.data, sample_size, cudaMemcpyHostToDevice);
                        downloaded_queue_.push(pair.second);
                        offset += sample_size;
                    }

                    downloaded_ = true;
                    inference_mutex_.unlock();
                    
                    if (poisoned)
                        break;
                }
            }
        }
    }

    void inference_thread() {
        bool poisoned = false;
        while (!poisoned)  {
            inference_mutex_.lock();
            if (downloaded_) {
                BoundingBox* boxes = inference_call_(model_ptr_, n_boxes_);
                
                size_t loaded = 0;
                size_t offset = 0;
                while(downloaded_queue_.size() > 0)
                {
                    auto frame = downloaded_queue_.front();
                    downloaded_queue_.pop();
                    auto bboxes = load_bboxes(boxes + offset, n_boxes_[loaded]);
                    offset += n_boxes_[loaded++];

                    process_mutex_.lock();
                    processed_queue_.push(
                        std::make_pair(frames, bboxes)
                    );
                    process_mutex_.unlock();

                }
                downloaded_ = false;
                process_mutex_.lock();

                if (loaded < batchsize) {
                    poisoned = true;

                    std::vector<BoundingBox> bboxes;
                    process_mutex_.lock();
                    processed_queue_.push(
                        std::make_pair(nullptr, bboxes)
                    );
                    process_mutex_.unlock();
                }
                
            } else {
                inference_mutex_.unlock();
                freeze_thread(INFERENCE_RETRY_MS);
            }
        }
    }

    void process_thread() {
        bool poisoned = false;
        while (!poisoned) {

        }
    }



private:
    DataLoader* loader_;
    DataProcesser* processer_;
    size_t batchsize_;
    size_t width_;
    size_t height_;
    void* inp_ptr_;
    void* model_ptr_;

    size_t *n_boxes_;
    BoundingBox* (*inference_call_)(void* model, size_t **n_boxes);
    
    Queue<cv::Mat*> loaded_queue_;
    Queue<cv::Mat*> downloaded_queue_;
    bool downloaded_;
    Queue<std::pair<cv::Mat*,std::vector<BoundingBox>>> processed_queue_;



};