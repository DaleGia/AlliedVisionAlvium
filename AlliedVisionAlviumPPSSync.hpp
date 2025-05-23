#ifndef ALLIEDVISIONALVIUMSYNC_H_
#define ALLIEDVISIONALVIUMSYNC_H_

#include <opencv2/opencv.hpp>
#include "VmbCPP/VmbCPP.h"
#include "VmbImageTransform/VmbTransform.h"
#include "AlliedVisionAlvium.hpp"
#include "PPSSync.hpp"
#include "GNSS.hpp"

class AlliedVisionAlviumPPSSynchronisedFrameData
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

    int64_t systemTimestampAtLastPPS = 0;
    int64_t cameraTimestampAtLastPPS = 0;
    int64_t systemJitterAtLastPPS = 0;
    int64_t cameraJitterAtLastCameraPPS = 0;

    int64_t lastGNSStimestamp = 0;
    double lastGNSSLatitude = 0.0;
    double lastGNSSLongitude = 0.0;
    double lastGNSSAltitudeMSL = 0.0;
    cv::Mat image;
};

class PPSSyncronisedFrameObserver : public VmbCPP::IFrameObserver
{
public:
    PPSSyncronisedFrameObserver(VmbCPP::CameraPtr camera) : IFrameObserver(camera) {};

    PPSSyncronisedFrameObserver(
        VmbCPP::CameraPtr camera,
        std::function<void(AlliedVisionAlviumPPSSynchronisedFrameData &, void *)> imageCallback,
        PPSSync *pps,
        GNSS *gnss,
        void *arg)
        : IFrameObserver(camera),
          pps(pps),
          gnss(gnss),
          callback(imageCallback),
          argument(arg) {
          };

    void FrameReceived(const VmbCPP::FramePtr frame);

private:
    std::function<void(AlliedVisionAlviumPPSSynchronisedFrameData &, void *)> callback = nullptr;
    void *argument = nullptr;
    PPSSync *pps = nullptr;
    GNSS *gnss = nullptr;
};

class AlliedVisionAlviumPPSSync : public AlliedVisionAlvium
{
public:
    AlliedVisionAlviumPPSSync();
    ~AlliedVisionAlviumPPSSync();

    bool enableSync(AlliedVisionAlvium::Line line);
    bool startAcquisition(
        int bufferCount,
        std::function<void(AlliedVisionAlviumPPSSynchronisedFrameData &, void *)> newFrameCallback,
        void *arg);

    bool getSingleSyncedFrame(AlliedVisionAlviumPPSSynchronisedFrameData &buffer, uint32_t timeoutMs);

private:
    static void cameraPPSCallback(
        int64_t cameraPPSTimestamp,
        int64_t systemPPSTimestamp,
        int64_t cameraPPSJitter,
        int64_t systemPPSJitter,
        void *arg);

    PPSSync ppsSync;
    GNSS gnss;
    std::mutex gnssMutex;
    int64_t lastGNSSTimestamp = 0.0;
    double lastGNSSLatitude = 0.0;
    double lastGNSSLongitude = 0.0;
};

#endif