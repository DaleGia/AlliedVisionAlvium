#include <opencv2/opencv.hpp>
#include "VmbCPP/VmbCPP.h"
#include "VmbImageTransform/VmbTransform.h"

class AlliedVisionAlviumFrameData
{
public:
    uint32_t height;
    uint32_t width;
    uint32_t offsetX;
    uint32_t offsetY;
    uint64_t frameId;
    uint64_t timestamp;
    time_t systemTimeSec;
    long systemTimeNSec;

    double exposureTime;
    double gain;

    cv::Mat image;
};
/**
 * \brief IFrameObserver implementation for asynchronous image acquisition
 */
class FrameObserver : public VmbCPP::IFrameObserver
{
public:
    FrameObserver(VmbCPP::CameraPtr camera) : IFrameObserver(camera) {};

    FrameObserver(
        VmbCPP::CameraPtr camera,
        std::function<void(AlliedVisionAlviumFrameData &, void *)> imageCallback,
        void *arg) : IFrameObserver(camera), callback(imageCallback), argument(arg) {

                     };

    void FrameReceived(const VmbCPP::FramePtr frame);

private:
    static VmbErrorType GetFeatureValueAsString(VmbCPP::FeaturePtr feat, std::string &val);

    std::function<void(AlliedVisionAlviumFrameData &, void *)> callback = nullptr;
    void *argument = nullptr;
};

/**
 * \brief IFrameObserver implementation for asynchronous image acquisition
 */
class EventObserver : public VmbCPP::IFeatureObserver
{
public:
    EventObserver() : VmbCPP::IFeatureObserver() {};

    EventObserver(
        std::function<void(std::string, uint64_t, time_t, long, void *)> eventCallback,
        void *arg) : VmbCPP::IFeatureObserver(), callback(eventCallback), argument(arg) {};

    void FeatureChanged(const VmbCPP::FeaturePtr &feature);

private:
    std::function<void(std::string, uint64_t, time_t, long, void *)> callback = nullptr;
    void *argument = nullptr;
};

class AlliedVisionAlvium
{
public:
    AlliedVisionAlvium();
    ~AlliedVisionAlvium();

    bool connect();
    bool connectByName(std::string cameraName);
    std::vector<std::string> getNames(void);

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
        std::function<void(AlliedVisionAlviumFrameData &, void *)> newFrameCallback,
        void *arg);

    bool stopAcquisition(void);

    bool getSingleFrame(AlliedVisionAlviumFrameData &buffer, uint32_t timeoutMs);

    bool setDeviceThroughputLimit(std::string buffer);

private:
    static VmbErrorType GetFeatureValueAsString(VmbCPP::FeaturePtr feat, std::string &val);

    bool getCameraNameFromDeviceIdList(std::string deviceID, std::string &cameraName);

    VmbCPP::CameraPtr camera;
    bool cameraOpen = false;
};
