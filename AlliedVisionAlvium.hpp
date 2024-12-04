#include <opencv2/opencv.hpp>
#include "VmbCPP/VmbCPP.h"
#include "VmbImageTransform/VmbTransform.h"

/**
 * \brief IFrameObserver implementation for asynchronous image acquisition
 */
class FrameObserver : public VmbCPP::IFrameObserver
{
public:
    FrameObserver(VmbCPP::CameraPtr camera) : IFrameObserver(camera) {
                                              };

    FrameObserver(
        VmbCPP::CameraPtr camera,
        std::function<void(cv::Mat, uint64_t, uint64_t, void *)> imageCallback,
        void *arg) : IFrameObserver(camera), callback(imageCallback), argument(arg) {

                     };

    void FrameReceived(const VmbCPP::FramePtr frame);

private:
    std::function<void(cv::Mat, uint64_t, uint64_t, void *)> callback = nullptr;
    void *argument = nullptr;
};

/**
 * \brief IFrameObserver implementation for asynchronous image acquisition
 */
class EventObserver : public VmbCPP::IFeatureObserver
{
public:
    EventObserver() : VmbCPP::IFeatureObserver() {
                      };

    EventObserver(
        std::function<void(std::string, uint64_t, time_t, time_t, void *)> eventCallback,
        void *arg) : VmbCPP::IFeatureObserver(), callback(eventCallback), argument(arg) {
                     };

    void FeatureChanged(const VmbCPP::FeaturePtr &feature);

private:
    std::function<void(std::string, uint64_t, time_t, time_t, void *)> callback = nullptr;
    void *argument = nullptr;
};

class AlliedVisionAlvium
{
public:
    AlliedVisionAlvium();
    ~AlliedVisionAlvium();

    bool connect();
    bool connectByDeviceID(std::string deviceID);

    bool disconnect(void);
    bool isCameraOpen(void);

    std::string getName(void);

    bool getFeature(
        std::string featureName,
        std::string &featureValue);
    bool setFeature(
        std::string featureName,
        std::string featureValue);

    bool activateEvent(
        std::string eventName,
        std::string eventObserverName,
        std::function<void(
            std::string,
            int64_t,
            time_t,
            time_t,
            void *)>
            eventCallback,
        void *arg);

    bool runCommand(std::string command);

    bool startAcquisition(
        int bufferCount,
        std::function<void(cv::Mat, uint64_t, uint64_t, void *)> newFrameCallback,
        void *arg);
    bool stopAcquisition(void);

    bool getSingleFrame(cv::Mat &buffer, uint64_t &cameraFrameID, uint64_t &cameraTimestamp, uint32_t timeoutMs);

    bool setDeviceThroughputLimit(std::string buffer);

private:
    bool getCameraNameFromDeviceIdList(std::string deviceID, std::string &cameraName);

    VmbCPP::CameraPtr camera;
    bool cameraOpen = false;
};
