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
    FrameObserver(VmbCPP::CameraPtr camera) : IFrameObserver(camera) {
                                              };

    FrameObserver(
        VmbCPP::CameraPtr camera,
        std::function<void(AlliedVisionAlviumFrameData &, void *)> imageCallback,
        void *arg) : IFrameObserver(camera), callback(imageCallback), argument(arg) {

                     };

    void FrameReceived(const VmbCPP::FramePtr frame);

private:
    std::function<void(AlliedVisionAlviumFrameData &, void *)> callback = nullptr;
    void *argument = nullptr;

    VmbErrorType GetFeatureValueAsString(VmbCPP::FeaturePtr feat, std::string &val)
    {
        VmbErrorType err;
        VmbFeatureDataType type;

        err = feat->GetDataType(type);

        if (err != VmbErrorSuccess)
        {
            return err;
        }

        switch (type)
        {
        case VmbFeatureDataBool:
        {
            VmbBool_t boolVal;
            if (feat->GetValue(boolVal) == VmbErrorSuccess)
            {
                val = boolVal ? "true" : "false";
                return VmbErrorSuccess;
            }
            break;
        }
        case VmbFeatureDataInt:
        {
            VmbInt64_t intVal;
            if (feat->GetValue(intVal) == VmbErrorSuccess)
            {
                val = std::to_string(intVal);
                return VmbErrorSuccess;
            }
            break;
        }
        case VmbFeatureDataFloat:
        {
            double floatVal;
            if (feat->GetValue(floatVal) == VmbErrorSuccess)
            {
                val = std::to_string(floatVal);
                return VmbErrorSuccess;
            }
            break;
        }
        case VmbFeatureDataEnum:
        case VmbFeatureDataString:
        {
            std::string stringVal;
            if (feat->GetValue(stringVal) == VmbErrorSuccess)
            {
                val = stringVal;
                return VmbErrorSuccess;
            }
            break;
        }
        default:
            break;
        }

        return VmbErrorNotSupported;
    }
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
        std::function<void(std::string, uint64_t, time_t, long, void *)> eventCallback,
        void *arg) : VmbCPP::IFeatureObserver(), callback(eventCallback), argument(arg) {
                     };

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
        std::function<void(AlliedVisionAlviumFrameData &, void *)> newFrameCallback,
        void *arg);

    bool stopAcquisition(void);

    bool getSingleFrame(cv::Mat &buffer, uint64_t &cameraFrameID, uint64_t &cameraTimestamp, uint32_t timeoutMs);

    bool setDeviceThroughputLimit(std::string buffer);

private:
    bool getCameraNameFromDeviceIdList(std::string deviceID, std::string &cameraName);

    VmbCPP::CameraPtr camera;
    bool cameraOpen = false;
};
