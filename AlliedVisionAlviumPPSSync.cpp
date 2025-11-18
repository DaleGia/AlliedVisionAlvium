#include "AlliedVisionAlviumPPSSync.hpp"
#include <iostream>
#include <VmbCPP/Frame.hpp>
#include <VmbCPP/Frame.h>

VmbError_t VMB_CALL ChunkCallbackSync(VmbHandle_t featureAccessHandle, void *userContext)
{
    AlliedVisionAlviumPPSSynchronisedFrameData *frameData;
    frameData = (AlliedVisionAlviumPPSSynchronisedFrameData *)userContext;

    VmbError_t err = VmbErrorSuccess;
    VmbInt64_t ts = 0, w = 0, h = 0;

    err = VmbFeatureFloatGet(featureAccessHandle, "ChunkExposureTime", &frameData->exposureTimeUs);
    if (err != VmbErrorSuccess)
    {
        std::cerr << "Could not get exposure time from chunk data" << std::endl;
        return err;
    }

    err = VmbFeatureFloatGet(featureAccessHandle, "ChunkGain", &frameData->gainDb);
    if (err != VmbErrorSuccess)
    {
        std::cerr << "Could not get gain from chunk data" << std::endl;
        return err;
    }

    return err;
}

void PPSSyncronisedFrameObserver::FrameReceived(
    const VmbCPP::FramePtr frame)
{
    timespec ts;
    auto now = clock_gettime(CLOCK_REALTIME, &ts);

    AlliedVisionAlviumPPSSynchronisedFrameData frameData;

    this->pps->get(
        frameData.systemTimestampAtLastPPS,
        frameData.cameraTimestampAtLastPPS,
        frameData.systemJitterAtLastPPS,
        frameData.cameraJitterAtLastCameraPPS);

    this->gnss->get(
        frameData.lastGNSSLatitude,
        frameData.lastGNSSLongitude,
        frameData.lastGNSSAltitudeMSL,
        frameData.lastGNSStimestamp);

    VmbError_t err;
    int openCvType;
    VmbPixelFormatType format;
    VmbFrameStatusType status = VmbFrameStatusComplete;

    uint32_t bufferSize;
    uint8_t *data;
    /* These are used for unpacking images if need be */
    VmbImage sourceImage;
    VmbImage destinationImage;
    bool requiresUnpacking;

    err = frame->GetReceiveStatus(status);

    if (VmbErrorSuccess != err)
    {
        std::cerr << "AlliedVisionAlvium: Could not get frame status" << std::endl;
        m_pCamera->QueueFrame(frame);
        return;
    }
    else if (VmbFrameStatusComplete != status)
    {
        switch (status)
        {
        case VmbFrameStatusIncomplete:
        {
            std::cerr << "AlliedVisionAlvium: Frame incomplete. Try a slower frame rate" << std::endl;
            m_pCamera->QueueFrame(frame);
            return;
        }
        case VmbFrameStatusTooSmall:
        {
            std::cerr << "AlliedVisionAlvium: Frame too small..." << std::endl;
            m_pCamera->QueueFrame(frame);
            return;
        }
        case VmbFrameStatusInvalid:
        {
            std::cerr << "AlliedVisionAlvium: Frame invalid..." << std::endl;
            m_pCamera->QueueFrame(frame);
            return;
        }
        default:
        {
            std::cerr << "AlliedVisionAlvium: Frame status unknown..." << std::endl;
            m_pCamera->QueueFrame(frame);
            return;
        }
        }
    }

    /* Get all of the information about the frame including the chunk data */
    VmbUint64_t cameraTimestamp;
    VmbUint64_t frameID;

    frame->GetPixelFormat(format);
    frame->GetBufferSize(bufferSize);
    frame->GetBuffer(data);
    frame->GetTimestamp(cameraTimestamp);
    frame->GetFrameID(frameID);
    frame->GetHeight(frameData.height);
    frame->GetWidth(frameData.width);
    frame->GetOffsetX(frameData.offsetX);
    frame->GetOffsetY(frameData.offsetY);
    frameData.systemImageReceivedTimestamp =
        ts.tv_sec * 10000000000 + ts.tv_nsec;
    frameData.cameraFrameStartTimestamp = cameraTimestamp;
    frameData.cameraFrameId = frameID;

    VmbFrame_t *frame2GetChunk = nullptr;
    err = frame->GetFrameStruct(frame2GetChunk);
    if (err != VmbErrorSuccess)
    {
        std::cerr << "Could not get frame struct:" << err << std::endl;
    }

    err = VmbChunkDataAccess(frame2GetChunk, ChunkCallbackSync, &frameData);
    if (err != VmbErrorSuccess)
    {
        std::cerr << "Could not access frame ChunkData:" << err << std::endl;
    }

    switch (format)
    {
    case VmbPixelFormatMono8:
    {
        openCvType = CV_8UC1;
        frameData.image = cv::Mat(
                              frameData.height,
                              frameData.width,
                              openCvType,
                              data)
                              .clone();
        break;
    }
    case VmbPixelFormatMono10:
    {
        openCvType = CV_16UC1;
        frameData.image = cv::Mat(
                              frameData.height,
                              frameData.width,
                              openCvType,
                              data)
                              .clone();
        break;
    }
    case VmbPixelFormatMono12:
    {
        openCvType = CV_16UC1;
        frameData.image = cv::Mat(
                              frameData.height,
                              frameData.width,
                              openCvType,
                              data)
                              .clone();
        break;
    }
    case VmbPixelFormatMono12p:
    {
        openCvType = CV_16UC1;

        /* Convert the image to 16 bit*/
        sourceImage.Size = sizeof(sourceImage);
        /* The 2 is because it needs to fit 16 bit*/
        sourceImage.Data = data;
        destinationImage.Size = sizeof(destinationImage);
        destinationImage.Data = malloc(frameData.width * frameData.height * 2);
        if (nullptr == destinationImage.Data)
        {
            std::cerr << "Could not create destination buffer for unpacking" << std::endl;
            m_pCamera->QueueFrame(frame);
            return;
        }

        VmbError_t error = VmbSetImageInfoFromPixelFormat(
            VmbPixelFormatMono12p,
            frameData.width,
            frameData.height,
            &sourceImage);
        if (VmbErrorSuccess != error)
        {
            std::cerr << "Could not create source image info for unpacking: " << error << std::endl;
            free(destinationImage.Data);
            m_pCamera->QueueFrame(frame);

            return;
        }
        error = VmbSetImageInfoFromInputParameters(
            VmbPixelFormatMono12,
            frameData.width,
            frameData.height,
            VmbPixelLayoutMono,
            16,
            &destinationImage);
        if (error != VmbErrorSuccess)
        {
            std::cerr << "Could not create destination image info for unpacking: " << error << std::endl;
            free(destinationImage.Data);
            m_pCamera->QueueFrame(frame);
        }
        error = VmbImageTransform(
            &sourceImage,
            &destinationImage,
            NULL,
            0);
        if (error != VmbErrorSuccess)
        {
            std::cerr << "Could not unpack image: " << error << std::endl;
            free(destinationImage.Data);
            m_pCamera->QueueFrame(frame);
        }
        frameData.image = cv::Mat(
                              frameData.height,
                              frameData.width,
                              openCvType,
                              destinationImage.Data)
                              .clone();
        free(destinationImage.Data);
        break;
    }
    default:
    {
        std::cerr << "Camera frame format not supported... " << format << std::endl;
        m_pCamera->QueueFrame(frame);
        return;
    }
    }

    /* returns the frame buffer back to the queue */
    m_pCamera->QueueFrame(frame);
    if (nullptr != this->callback)
    {
        this->callback(frameData, this->argument);
    }
};

AlliedVisionAlviumPPSSync::AlliedVisionAlviumPPSSync()
{
    this->gnss.start();
}

AlliedVisionAlviumPPSSync::~AlliedVisionAlviumPPSSync()
{
}

bool AlliedVisionAlviumPPSSync::getSingleSyncedFrame(
    AlliedVisionAlviumPPSSynchronisedFrameData &buffer,
    uint32_t timeoutMs)
{
    VmbCPP::FramePtr frame;
    VmbError_t err;
    int openCvType;
    VmbPixelFormatType format;
    VmbFrameStatusType status;
    uint32_t bufferSize;
    cv::Mat image;
    uint8_t *data;
    VmbUint64_t timestamp;
    VmbUint64_t frameID;
    /* These are used for unpacking images if need be */
    VmbImage sourceImage;
    VmbImage destinationImage;
    bool requiresUnpacking;

    AlliedVisionAlviumFrameData frameData;

    if (false == this->AlliedVisionAlvium::getSingleFrame(frameData, 12000))
    {
        return false;
    }

    this->ppsSync.get(
        buffer.systemTimestampAtLastPPS,
        buffer.cameraTimestampAtLastPPS,
        buffer.systemJitterAtLastPPS,
        buffer.cameraJitterAtLastCameraPPS);

    this->gnss.get(
        buffer.lastGNSSLatitude,
        buffer.lastGNSSLongitude,
        buffer.lastGNSSAltitudeMSL,
        buffer.lastGNSStimestamp);

    buffer.cameraFrameStartTimestamp = frameData.cameraFrameStartTimestamp;
    buffer.cameraFrameId = frameData.cameraFrameId;
    buffer.systemImageReceivedTimestamp = frameData.systemImageReceivedTimestamp;
    buffer.exposureTimeUs = frameData.exposureTimeUs;
    buffer.gainDb = frameData.gainDb;
    buffer.height = frameData.height;
    buffer.width = frameData.width;
    buffer.offsetX = frameData.offsetX;
    buffer.offsetY = frameData.offsetY;
    buffer.image = frameData.image;

    return true;
}
bool AlliedVisionAlviumPPSSync::enableSync(AlliedVisionAlvium::Line line)
{
    if (false == this->ppsSync.enable(this, line, cameraPPSCallback, nullptr))
    {
        std::cerr << "Unable to enable PPS Sync" << std::endl;
        return false;
    }

    return true;
}

void AlliedVisionAlviumPPSSync::cameraPPSCallback(
    int64_t cameraPPSTimestamp,
    int64_t systemPPSTimestamp,
    int64_t cameraPPSJitter,
    int64_t systemPPSJitter,
    void *arg)
{

    time_t seconds = systemPPSTimestamp / 1000000000LL;
    time_t milliseconds = (systemPPSTimestamp % 1000000000LL) / 1000000LL;
    if (milliseconds > 700)
    {
        seconds += 1;
    }
}

bool AlliedVisionAlviumPPSSync::startAcquisition(
    int bufferCount,
    std::function<void(
        AlliedVisionAlviumPPSSynchronisedFrameData &,
        void *)>
        newFrameCallback,
    void *arg)
{
    VmbErrorType err;
    err = camera->StartContinuousImageAcquisition(
        bufferCount,
        VmbCPP::IFrameObserverPtr(new PPSSyncronisedFrameObserver(
            camera,
            newFrameCallback,
            &this->ppsSync,
            &this->gnss,
            arg)));
    if (VmbErrorSuccess != err)
    {
        std::cerr << "Unable to start image Aqcuisition... " << err << std::endl;
        return false;
    }

    return true;
}
