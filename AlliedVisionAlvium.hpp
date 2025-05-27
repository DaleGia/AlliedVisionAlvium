#ifndef ALLIEDVISIONALVIUM_H_
#define ALLIEDVISIONALVIUM_H_

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
    uint64_t cameraFrameId;
    uint64_t cameraFrameStartTimestamp;
    uint64_t systemImageReceivedTimestamp;

    double exposureTimeUs;
    double gainDb;

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

class AlliedVisionAlvium : public VmbCPP::ICameraListObserver
{
public:
    enum Line
    {
        LINE0 = 0,
        LINE1 = 1,
        LINE2 = 2,
        LINE3 = 3
    };

    AlliedVisionAlvium();
    ~AlliedVisionAlvium();

    bool connect();
    bool connectByUserId(std::string userId);
    std::vector<std::string> getUserIds(void);

    bool disconnect(void);
    bool isCameraOpen(void);

    std::string getName(void);
    std::string getUserId(void);
    bool getFeature(
        std::string featureName,
        std::string &featureValue);

    static VmbErrorType getFeature(VmbCPP::FeaturePtr feat, std::string &val);

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

    bool enableExternalTrigger(AlliedVisionAlvium::Line line);
    bool disableExternalTrigger(void);

protected:
    VmbCPP::CameraPtr camera;
    void CameraListChanged(
        VmbCPP::CameraPtr pCam,
        VmbCPP::UpdateTriggerType reason);

private:
    bool getCameraUserIdFromDeviceIdList(
        std::string cameraUserId,
        std::string &deviceID);


    bool cameraOpen = false;
};

#endif