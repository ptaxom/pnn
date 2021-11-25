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
        draw_bboxes(*image, boxes, class_names_);
        cv::imshow("demo", *image);
        char key = (char)cv::waitKey(1);
        return key == 'q';
    }

    DetectionRenderer(const std::vector<std::string> &class_names): DataProcesser(), class_names_(class_names) {}

private:
    std::vector<std::string> class_names_;
};

template<typename T>
class Queue{
public:
    Queue(const size_t max_size=0): max_size_(max_size), poisoned_(false) {}

    void push(const T &item) {
        std::unique_lock<std::mutex> lk(mutex_);
        if (max_size_ > 0) {
            cv_.wait(lk, [this]{return queue_.size() < max_size;});
        }
        queue_.push(item);
        lk.unlock();
        cv_.notify_one();
    }

    const T& pop() {
        std::unique_lock<std::mutex> lk(mutex_);
        cv_.wait(lk, [this]{return queue_.size() > 0;});
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
        return poisoned_ && queue_.size() == 0;
    }

    void set_limit(size_t max_size) {
        max_size_ = max_size;
    }

    void flush() {
        std::unique_lock<std::mutex> lk(mutex_);
        while (queue_.size()) queue_.pop();
        lk.unlock();
        cv_.notify_all();
    }

    std::queue<T> drain(const size_t n_elements) {
        std::unique_lock<std::mutex> lk(mutex_);
        cv_.wait(lk, [this]{return queue_.size() >= n_elements;});
        std::queue<T> items;
        for(size_t i = 0; i < n_elements; i++) {
            items.push(queue_.front())
            queue_.pop();
        }
        lk.unlock();
        cv_.notify_all();
        return items;
    }

private:
    size_t max_size_;
    bool poisoned_;
    std::queue<T> queue_;
    std::mutex mutex_;
    std::condition_variable cv_;
};


const size_t MAX_LOAD_QUEUE_SIZE = 40; // Bufferization up to 10 frames per sample in batch

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
        preprocess_finished_(false), exited_(false)
    {
        n_boxes_ = new size_t[batchsize];
        loaded_queue_.set_limit(batchsize * MAX_LOAD_QUEUE_SIZE);
        
    }

    void start() {
        load_worker_ = std::thread(&ThreadedProcesser::load_thread, this);
        download_worker_ = std::thread(&ThreadedProcesser::preprocess_thread, this);
        infer_worker_ = std::thread(&ThreadedProcesser::inference_thread, this);
        processer_worker_ = std::thread(&ThreadedProcesser::process_thread, this);
    }

    void join() {
        load_worker_.join();
        download_worker_.join();
        infer_worker_.join();
        processer_worker_.join();
    }

    ~ThreadedProcesser() {
        delete[] n_boxes_;
    }

private:
    

    static void freeze_thread(size_t ms) {
        if (ms <= 0)
            return;
        std::chrono::milliseconds timespan(ms);
        std::this_thread::sleep_for(timespan);
    }
    
    void load_thread() {
        while (!exited_) {
            auto frame = loader_->load();
            if (!frame) {
                loaded_queue_.poison();
                break;
            }
            loaded_queue_.push(frame);
        }
        if (exited_) {
            loaded_queue_.flush();
            loaded_queue_.poison();
        }
    }

    cv::Mat preprocess(cv::Mat *frame) {
        cv::Mat preprocessed_image;
        cv::resize(preprocessed_image, *frame, cv::Size(width_, height_));
        if (preprocessed_image.empty()) exit(101);

        preprocessed_image.convertTo(preprocessed_image, CV_32FC3, 1 / 255.0);
        if (preprocessed_image.empty()) exit(101);

        cv::Mat channels[3];
        cv::split(preprocessed_image, channels);
        std::swap(channels[0], channels[2]);
        cv::merge(channels, 3, preprocessed_image);
    }

    void preprocess_thread() {
        std::queue<std::pair<cv::Mat, cv::Mat*>> processed;
        bool poisoned = false;
        size_t sample_size = 3 * width_ * height_ * sizeof(float);

        while (!poisoned) {
            if (loaded_queue_.is_poisoned()) {
                poisoned = true;
            } else {
                auto frame = loaded_queue_.pop();
                processed.push(
                    std::make_pair(preprocess(frame), frame)
                );
            }
            
            if (poisoned || processed.size() >= batchsize_) {
                std::unique_lock<std::mutex> lk(inference_mutex_);
                inference_cv_.wait(lk, [this]{return !downloaded_;});
                size_t offset = 0;

                while(processed.size() > 0) {
                    auto pair = processed.front();
                    processed.pop();
                    cudaMemcpy(inp_ptr_ + offset, pair.first.data, sample_size, cudaMemcpyHostToDevice);
                    downloaded_queue_.push(pair.second);
                    offset += sample_size;
                }
                downloaded_ = true;
                if (poisoned)
                    preprocess_finished_ = true;
                lk.unlock();
                inference_cv_.notify_all();
            }
        }
    }

    void inference_thread() {
       while(true) {
            std::unique_lock<std::mutex> lk(inference_mutex_);
            inference_cv_.wait(lk, [this]{return downloaded_;});
            BoundingBox* boxes = inference_call_(model_ptr_, n_boxes_);
            
            size_t offset = 0, i = 0;
            while (downloaded_queue_.size()) {
                cv::Mat* frame = downloaded_queue_.front();
                downloaded_queue_.pop();
                auto bboxes = load_bboxes(n_boxes_[i], boxes + offset);
                processed_queue_.push(std::make_pair(frame, bboxes));
                offset += n_boxes_[i+1];
            }
            downloaded_ = false;
            lk.unlock();
            inference_cv_.notify_all();

            if (preprocess_finished_) {
                processed_queue_.poison();
                break;
            }
       }
    }

    void process_thread() {
        bool poisoned = false;
        while (!processed_queue_.is_poisoned()) {
            auto data = processed_queue_.pop();
            exited_ = processer_->postprocess(data.first, data.second);
            delete data.first;
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
    BoundingBox* (*inference_call_)(void* model, size_t *n_boxes);
    
    Queue<cv::Mat*> loaded_queue_;
    std::queue<cv::Mat*> downloaded_queue_;
    Queue<std::pair<cv::Mat*,std::vector<BoundingBox>>> processed_queue_;
    
    std::mutex inference_mutex_;
    std::condition_variable inference_cv_;
    bool downloaded_;
    bool preprocess_finished_;
    bool exited_;
    std::vector<std::thread&> workers;

    // ????
    std::thread load_worker_, download_worker_, infer_worker_, processer_worker_;
};

void visual_demo(const char* video_path, 
                 const char** c_classes, 
                 size_t batchsize,
                 size_t width,
                 size_t height,
                 void* inp_ptr,
                 void* model_ptr,
                 BoundingBox* (*infer_call)(void* model, size_t *n_boxes)
                 ) {
    DataLoader *loader = new VideoLoader(video_path);
    DataProcesser *processer = new DetectionRenderer(load_classes(c_classes));
    ThreadedProcesser worker(loader, processer, batchsize, width, height, inp_ptr, model_ptr, infer_call);
    worker.start();
    worker.join();
    delete processer;
    delete loader;
}