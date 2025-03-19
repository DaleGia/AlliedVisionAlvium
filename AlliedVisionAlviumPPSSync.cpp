#include "AlliedVisionAlviumPPSSync.hpp"
#include <iostream>

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
    }
    else if (VmbFrameStatusComplete != status)
    {
        switch (status)
        {
        case VmbFrameStatusIncomplete:
        {
            std::cerr << "AlliedVisionAlvium: Frame incomplete. Try a slower frame rate" << std::endl;
            return;
        }
        case VmbFrameStatusTooSmall:
        {
            std::cerr << "AlliedVisionAlvium: Frame too small..." << std::endl;
            return;
        }
        case VmbFrameStatusInvalid:
        {
            std::cerr << "AlliedVisionAlvium: Frame invalid..." << std::endl;
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

    // Access the Chunk data of the incoming frame. Chunk data accesible inside lambda function
    err = frame->AccessChunkData(
        [this, &frameData](VmbCPP::ChunkFeatureContainerPtr &chunkFeatures) -> VmbErrorType
        {
            VmbCPP::FeaturePtr feat;
            VmbErrorType err;

            // Get a specific Chunk feature via the FeatureContainer chunkFeatures
            err = chunkFeatures->GetFeatureByName("ExposureTime", feat);
            if (err != VmbErrorSuccess)
            {
                std::cerr << "Could not get Exposure time from frame ChunkData" << std::endl;
            }

            // The Chunk feature can be read like any other feature
            std::string val;
            err = AlliedVisionAlvium::getFeature(feat, val);
            if (err != VmbErrorSuccess)
            {
                std::cerr << "Could not get Exposure feature value as string from frame ChunkData" << std::endl;
            }
            else
            {
                frameData.exposureTimeUs = std::stod(val);
            }

            // Get a specific Chunk feature via the FeatureContainer chunkFeatures
            err = chunkFeatures->GetFeatureByName("Gain", feat);
            if (err != VmbErrorSuccess)
            {
                std::cerr << "Could not get Gain from frame ChunkData" << std::endl;
            }

            // The Chunk feature can be read like any other feature
            val = "";
            err = AlliedVisionAlvium::getFeature(feat, val);
            if (err != VmbErrorSuccess)
            {
                std::cerr << "Could not get Gain feature value as string from frame ChunkData" << std::endl;
            }
            else
            {
                frameData.gainDb = std::stod(val);
            }

            return VmbErrorSuccess;
        });

    switch (format)
    {
    case VmbPixelFormatMono8:
    {
        openCvType = CV_8UC1;
        frameData.image = cv::Mat(
            frameData.height,
            frameData.width,
            openCvType,
            data);
        break;
    }
    case VmbPixelFormatMono10:
    {
        openCvType = CV_16UC1;
        frameData.image = cv::Mat(
            frameData.height,
            frameData.width,
            openCvType,
            data);
        break;
    }
    case VmbPixelFormatMono12:
    {
        openCvType = CV_16UC1;
        frameData.image = cv::Mat(
            frameData.height,
            frameData.width,
            openCvType,
            data);
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
        }
        error = VmbImageTransform(
            &sourceImage,
            &destinationImage,
            NULL,
            0);
        if (error != VmbErrorSuccess)
        {
            std::cerr << "Could not unpack image: " << error << std::endl;
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

bool AlliedVisionAlviumPPSSync::getSingleFrame(
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
        std::cerr << "Could not get single frame: " << err << std::endl;
        return false;
    }

    timespec ts;
    auto now = clock_gettime(CLOCK_REALTIME, &ts);
    this->ppsSync.get(
        buffer.systemTimestampAtLastPPS,
        buffer.cameraTimestampAtLastPPS,
        buffer.systemJitterAtLastPPS,
        buffer.cameraJitterAtLastCameraPPS);

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
    buffer.exposureTimeUs = frameData.exposureTimeUs;
    buffer.gainDb = frameData.gainDb;

    return true;
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
