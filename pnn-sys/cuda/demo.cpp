#include "kernels.h"

#include <queue>
#include <chrono>
#include <thread>
#include <condition_variable>
#include <memory>


class Stopwatch {
public:
    void tick() {
        start_ = std::chrono::steady_clock::now();
    }

    void tack() {
        end_ = std::chrono::steady_clock::now();
    }

    double duration() {
        return std::chrono::duration_cast<std::chrono::microseconds>(end_ - start_).count() / 1000.;
    }

private:
    std::chrono::steady_clock::time_point start_;
    std::chrono::steady_clock::time_point end_;
};


struct Task {
    size_t id;
    cv::Mat original;
    std::vector<BoundingBox> boxes;
};

using TaskPtr = std::unique_ptr<Task>;

class DataLoader {
public:
    virtual TaskPtr load() = 0;
    
    virtual size_t size() = 0;
};

class VideoLoader: public DataLoader {
public:
    virtual TaskPtr load() {
        cv::Mat frame;
        if (!capture_.read(frame)) {
            return nullptr;
        }
        loaded_++;
        return std::unique_ptr<Task>(new Task{loaded_, std::move(frame), std::vector<BoundingBox>()});
    }

    VideoLoader(const std::string &video_path) : DataLoader() {
        capture_ = cv::VideoCapture(video_path);
        if (!capture_.isOpened()) 
            throw std::runtime_error("Couldnt open video");
    }

    virtual size_t size(){
        return loaded_;
    }

    ~VideoLoader() {
        capture_.release();
    }

private:
    cv::VideoCapture capture_;
    size_t loaded_ = 0;
};

class DataProcesser {
public:
    virtual bool postprocess(cv::Mat &image, std::vector<BoundingBox> &boxes) = 0;
};

class DetectionRenderer : public DataProcesser {
public:
    virtual bool postprocess(cv::Mat &image, std::vector<BoundingBox> &boxes) {
        draw_bboxes(image, boxes, class_names_);
        cv::imshow("demo", image);
        char key = (char)cv::waitKey(1);
        return key == 'q';
    }

    DetectionRenderer(const std::vector<std::string> &class_names): DataProcesser(), class_names_(class_names) {}

private:
    std::vector<std::string> class_names_;
};

class FileRenderer : public DataProcesser {
public:
    virtual bool postprocess(cv::Mat image, std::vector<BoundingBox> &boxes) {
        draw_bboxes(image, boxes, class_names_);
        if (!writer_) {
            writer_ = new cv::VideoWriter("../models/demo.avi", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 25, image.size());
            if (!writer_ || !writer_->isOpened()) {
                return true;
            }
        }
        writer_->write(image);
        return false;
    }

    FileRenderer(const std::vector<std::string> &class_names): DataProcesser(), class_names_(class_names) {}
    
    ~FileRenderer() {
        if (writer_) {
            writer_->release();
            delete writer_;
        }
    }

private:
    std::vector<std::string> class_names_;
    cv::VideoWriter *writer_ = nullptr;
};

template<typename T>
class Queue{
public:
    Queue(const size_t max_size=0): max_size_(max_size), poisoned_(false) {}

    void push(T &item) {
        std::unique_lock<std::mutex> lk(mutex_);
        if (max_size_ > 0) {
            cv_.wait(lk, [this]{return queue_.size() < max_size_;});
        }
        queue_.push(std::move(item));
        lk.unlock();
        cv_.notify_one();
    }

    T pop() {
        std::unique_lock<std::mutex> lk(mutex_);
        cv_.wait(lk, [this]{return queue_.size() > 0 || poisoned();});
        if (poisoned()) 
            throw std::runtime_error("Waiting on poisoned queue");
        T item = std::move(queue_.front());
        queue_.pop();
        lk.unlock();
        cv_.notify_one();
        return item;
    }

    void poison() {
        std::unique_lock<std::mutex> lk(mutex_);
        poisoned_ = true;
        lk.unlock();
        cv_.notify_all();
    }

    bool is_poisoned() {
        std::lock_guard<std::mutex> lock(mutex_);
        return poisoned();
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
        cv_.wait(lk, [=]{return queue_.size() >= n_elements;});
        std::queue<T> items;
        for(size_t i = 0; i < n_elements; i++) {
            items.push(queue_.front());
            queue_.pop();
        }
        lk.unlock();
        cv_.notify_all();
        return items;
    }

private:
    bool poisoned() {
        return poisoned_ && queue_.size() == 0;
    }
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
                      BoundingBox* (*inference_call)(void* model, size_t *n_boxes, double *infer_time)
                      ): 
        loader_(loader), processer_(processer), batchsize_(batchsize),
        width_(width), height_(height), inp_ptr_(static_cast<float*>(inp_ptr)),
        downloaded_(false), model_ptr_(model_ptr),
        inference_call_(inference_call),
        preprocess_finished_(false), exited_(false)
    {
        n_boxes_ = new size_t[batchsize];
        loaded_queue_.set_limit(batchsize * MAX_LOAD_QUEUE_SIZE);
        
    }

    void start() {
        total_watcher_.tick();
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
        total_watcher_.tack();
    }

    ~ThreadedProcesser() {
        delete[] n_boxes_;
    }

    InferStats stats() {
        InferStats stats;
        stats.total_frames = loader_->size();
        stats.inference_time = inference_time_;
        stats.inference_with_nms = nms_time_;
        stats.total_time = total_watcher_.duration();
        return stats;
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

    std::pair<TaskPtr, std::vector<cv::Mat>> preprocess(TaskPtr &task) {
        cv::Mat preprocessed_image;
        cv::resize(task->original, preprocessed_image, cv::Size(width_, height_));
        if (preprocessed_image.empty()) exit(101);

        preprocessed_image.convertTo(preprocessed_image, CV_32FC3, 1 / 255.0);
        if (preprocessed_image.empty()) exit(101);

        std::vector<cv::Mat> channels;
        cv::split(preprocessed_image, channels);
        return std::make_pair(std::move(task), channels);
    }

    void preprocess_thread() {
        std::queue<std::pair<TaskPtr, std::vector<cv::Mat>>> processed;
        bool poisoned = false;
        size_t channel_size = width_ * height_;

        while (!poisoned) {
            if (loaded_queue_.is_poisoned()) {
                poisoned = true;
            } else {
                TaskPtr task;
                try {
                    task = loaded_queue_.pop();
                } catch (...) {
                    poisoned = true;
                }
                processed.push(preprocess(task));
            }
            
            if (poisoned || processed.size() >= batchsize_) {
                std::unique_lock<std::mutex> lk(inference_mutex_);
                inference_cv_.wait(lk, [this]{return !downloaded_;});
                size_t offset = 0;

                while(processed.size() > 0) {
                    auto pair = std::move(processed.front());
                    processed.pop();
                    for(int channel = 0; channel < 3; channel++)
                    {
                        cudaMemcpy(inp_ptr_ + offset, pair.second[2 - channel].data, channel_size  * sizeof(float), cudaMemcpyHostToDevice);
                        offset += channel_size;
                    }
                    downloaded_queue_.push(std::move(pair.first));
                }
                downloaded_ = true;
                if (poisoned) {
                    preprocess_finished_ = true;
                }
                lk.unlock();
                inference_cv_.notify_all();
            }
        }
    }

    void inference_thread() {
       while(true) {
            std::unique_lock<std::mutex> lk(inference_mutex_);
            inference_cv_.wait(lk, [this]{return downloaded_;});
            double infer_time;
            Stopwatch nms_watcher;

            nms_watcher.tick();
            BoundingBox* boxes = inference_call_(model_ptr_, n_boxes_, &infer_time);
            nms_watcher.tack();

            inference_time_ += infer_time * 1000.;
            nms_time_ += nms_watcher.duration();
            
            size_t offset = 0, i = 0;
            while (downloaded_queue_.size()) {
                auto task = std::move(downloaded_queue_.front());
                downloaded_queue_.pop();
                task->boxes = load_bboxes(n_boxes_[i], boxes + offset);
                processed_queue_.push(task);
                offset += n_boxes_[i+1];
            }
            free(boxes);
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
        while (!processed_queue_.is_poisoned()) {
            TaskPtr task;
            try {
                task = std::move(processed_queue_.pop());
            } catch (...) {
                exited_ = true;
                break;
            }
            exited_ = processer_->postprocess(task->original, task->boxes);
            task.reset();
            if (exited_)
                break;
        }
    }

private:
    DataLoader* loader_;
    DataProcesser* processer_;
    size_t batchsize_;
    size_t width_;
    size_t height_;
    float* inp_ptr_;
    void* model_ptr_;

    size_t *n_boxes_;
    BoundingBox* (*inference_call_)(void* model, size_t *n_boxes, double *infer_time);
    
    Queue<TaskPtr> loaded_queue_;
    std::queue<TaskPtr> downloaded_queue_;
    Queue<TaskPtr> processed_queue_;
    
    std::mutex inference_mutex_;
    std::condition_variable inference_cv_;
    bool downloaded_;
    bool preprocess_finished_;
    bool exited_;
    double inference_time_ = 0., nms_time_ = 0.;
    Stopwatch total_watcher_;

    // ????
    std::thread load_worker_, download_worker_, infer_worker_, processer_worker_;
};

InferStats visual_demo(const char* video_path, 
                 const char** c_classes, 
                 size_t batchsize,
                 size_t width,
                 size_t height,
                 void* inp_ptr,
                 void* model_ptr,
                 BoundingBox* (*infer_call)(void* model, size_t *n_boxes, double *infer_time)
                 ) {
    DataLoader *loader = new VideoLoader(video_path);
    DataProcesser *processer = new DetectionRenderer(load_classes(c_classes));
    ThreadedProcesser worker(loader, processer, batchsize, width, height, inp_ptr, model_ptr, infer_call);
    worker.start();
    worker.join();
    auto stats = worker.stats(); 
    delete processer;
    delete loader;
    return stats;
}