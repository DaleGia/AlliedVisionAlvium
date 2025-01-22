#include "AlliedVisionAlvium.hpp"
#include <iostream>

void FrameObserver::FrameReceived(const VmbCPP::FramePtr frame)
{
    timespec ts;
    auto now = clock_gettime(CLOCK_REALTIME, &ts);
    AlliedVisionAlviumFrameData frameData;
    frameData.systemTimeSec = ts.tv_sec;
    frameData.systemTimeNSec = ts.tv_nsec;

    VmbError_t err;
    int openCvType;
    VmbPixelFormatType format;
    VmbFrameStatusType status = VmbFrameStatusComplete;

    uint32_t bufferSize;
    uint8_t* data;
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

    frame->GetPixelFormat(format);

    frame->GetBufferSize(bufferSize);
    frame->GetBuffer(data);

    frame->GetHeight(frameData.height);
    frame->GetWidth(frameData.width);
    frame->GetOffsetX(frameData.offsetX);
    frame->GetOffsetY(frameData.offsetY);
    VmbUint64_t cameraTimestamp;
    frame->GetTimestamp(cameraTimestamp);
    frameData.timestamp = cameraTimestamp;
    VmbUint64_t frameID;
    frame->GetFrameID(frameID);
    // Access the Chunk data of the incoming frame. Chunk data accesible inside lambda function
    err = frame->AccessChunkData(
        [this, &frameData](VmbCPP::ChunkFeatureContainerPtr& chunkFeatures) -> VmbErrorType
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
            err = GetFeatureValueAsString(feat, val);
            if (err != VmbErrorSuccess)
            {
                std::cerr << "Could not get Exposure feature value as string from frame ChunkData" << std::endl;
            }
            else
            {
                frameData.exposureTime = std::stod(val);
            }

            // Get a specific Chunk feature via the FeatureContainer chunkFeatures
            err = chunkFeatures->GetFeatureByName("Gain", feat);
            if (err != VmbErrorSuccess)
            {
                std::cerr << "Could not get Gain from frame ChunkData" << std::endl;
            }

            // The Chunk feature can be read like any other feature
            val = "";
            err = GetFeatureValueAsString(feat, val);
            if (err != VmbErrorSuccess)
            {
                std::cerr << "Could not get Gain feature value as string from frame ChunkData" << std::endl;
            }
            else
            {
                frameData.gain = std::stod(val);
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

void EventObserver::FeatureChanged(const VmbCPP::FeaturePtr& feature)
{
    timespec ts;
    auto now = clock_gettime(CLOCK_REALTIME, &ts);
    VmbError_t err;
    std::string featureName("");
    VmbInt64_t featureValue;

    // Here an action can be perform based on the event that has occured
    if (feature == nullptr)
    {
        std::cerr << "EventObserver: null feature" << std::endl;
        return;
    }

    err = feature->GetName(featureName);
    if (VmbErrorSuccess != err)
    {
        std::cerr << "EventObserver: Could not get feature name" << std::endl;
        return;
    }

    err = feature->GetValue(featureValue);
    if (VmbErrorSuccess != err)
    {
        std::cerr << "EventObserver: Could not get feature value" << std::endl;
        return;
    }

    if (nullptr != this->callback)
    {
        this->callback(featureName, featureValue, ts.tv_sec, ts.tv_nsec, this->argument);
    }
}

AlliedVisionAlvium::AlliedVisionAlvium()
{}

AlliedVisionAlvium::~AlliedVisionAlvium()
{
    try
    {
        if (cameraOpen)
        {
            this->disconnect();
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << '\n';
    }
}

bool AlliedVisionAlvium::getCameraNameFromDeviceIdList(std::string cameraName, std::string& deviceID)
{
    VmbCPP::CameraPtrVector cameras;

    VmbCPP::VmbSystem& vimbax =
        VmbCPP::VmbSystem::GetInstance();

    VmbErrorType err;
    /* Start the API, get and open cameras */
    err = vimbax.Startup();
    if (err == VmbErrorAlready)
    {
        /* Do not do nothin'*/
    }
    else if (err != VmbErrorSuccess)
    {
        std::cerr << "AlliedVisionAlvium::getCameraList: Unable to start up vimbax " << std::endl;
        return false;
    }
    else if (VmbErrorSuccess != vimbax.GetCameras(cameras))
    {
        std::cerr << "AlliedVisionAlvium::getCameraList: Unable to detect any cameras" << std::endl;
        return false;
    }
    for (VmbCPP::CameraPtrVector::iterator iter = cameras.begin(); cameras.end() != iter; ++iter)
    {
        std::string name;
        std::string ID;
        if (VmbErrorSuccess != (*iter)->GetName(name))
        {
            std::cerr << "AlliedVisionAlvium::getCameraList: Unable to get camera name" << std::endl;
        }
        else if (name != cameraName)
        {
            continue;
        }
        else if (VmbErrorSuccess == (*iter)->GetID(deviceID))
        {
            return true;
        }
    }

    /* if we are up to here we have not found the right camera name */
    return false;
}

bool AlliedVisionAlvium::connect()
{

    VmbErrorType err;
    VmbCPP::CameraPtrVector cameras;
    std::string cameraID;
    VmbCPP::VmbSystem& vimbax =
        VmbCPP::VmbSystem::GetInstance();

    /* Start the API, get and open cameras */
    err = vimbax.Startup();
    if (err == VmbErrorAlready)
    {
        /* Do not do nothin'*/
    }
    else if (err != VmbErrorSuccess)
    {
        std::cerr << "Unable to start up vimbax : " << err << std::endl;
        this->cameraOpen = false;
        return false;
    }

    err = vimbax.GetCameras(cameras);
    if (err != VmbErrorSuccess)
    {
        std::cerr << "Unable to get camera list : " << err << std::endl;
        this->cameraOpen = false;
        return false;
    }
    else if (0 == cameras.size())
    {
        std::cerr << "No cameras detected: " << err << std::endl;
        this->cameraOpen = false;
        return false;
    }

    err = cameras[0]->GetID(cameraID);
    if (err != VmbErrorSuccess)
    {
        std::cerr << "Unable to get camera name : " << err << std::endl;
        this->cameraOpen = false;
        return false;
    }

    err = vimbax.OpenCameraByID(
        cameraID,
        VmbAccessModeExclusive,
        this->camera);
    if (err != VmbErrorSuccess)
    {
        std::cerr << "Unable to connect to camera " << cameraID << ": " << err << std::endl;
        this->cameraOpen = false;
        return false;
    }

    /* Enable the chunk data for different things so we can embedd them in the image data*/
    if (false == this->setFeature("ChunkEnable", "true"))
    {
        std::cerr << "Unable to set ChunkEnable to true" << std::endl;
    }

    if (false == this->setFeature("ChunkSelector", "ExposureTime"))
    {
        std::cerr << "Unable to set ChunkSelector to ExposureTime" << std::endl;
    }
    else if (false == this->setFeature("ChunkModeActive", "true"))
    {
        std::cerr << "Unable to set ChunkModeActive ExposureTime to true" << std::endl;
    }

    if (false == this->setFeature("ChunkSelector", "Gain"))
    {
        std::cerr << "Unable to set ChunkSelector to Gain" << std::endl;
    }
    else if (false == this->setFeature("ChunkModeActive", "true"))
    {
        std::cerr << "Unable to set ChunkModeActive Gain to true" << std::endl;
    }

    if (false == this->setFeature("ChunkSelector", "Gain"))
    {
        std::cerr << "Unable to set ChunkSelector to Gain" << std::endl;
    }
    else if (false == this->setFeature("ChunkModeActive", "true"))
    {
        std::cerr << "Unable to set ChunkModeActive Gain to true" << std::endl;
    }

    std::cout << "Connected to " << cameraID << std::endl;
    this->cameraOpen = true;
    return this->cameraOpen;
}

bool AlliedVisionAlvium::connectByDeviceID(std::string deviceID)
{

    VmbErrorType err;
    VmbCPP::VmbSystem& vimbax =
        VmbCPP::VmbSystem::GetInstance();

    /* Start the API, get and open cameras */
    err = vimbax.Startup();
    if (err == VmbErrorAlready)
    {
        /* Do not do nothin'*/
    }
    else if (err != VmbErrorSuccess)
    {
        std::cerr << "Unable to start up vimbax : " << err << std::endl;
        this->cameraOpen = false;
        return false;
    }

    std::string cameraName;
    if (false == this->getCameraNameFromDeviceIdList(deviceID, cameraName))
    {
        std::cerr << deviceID << " not detected..." << std::endl;
    }
    err = vimbax.OpenCameraByID(
        cameraName.c_str(),
        VmbAccessModeExclusive,
        this->camera);
    if (err != VmbErrorSuccess)
    {
        std::cerr << "Unable to connect to camera " << deviceID << ": " << err << std::endl;
        this->cameraOpen = false;
        return false;
    }

    this->cameraOpen = true;
    return this->cameraOpen;
}

bool AlliedVisionAlvium::disconnect(void)
{
    VmbErrorType err;
    this->stopAcquisition();
    err = this->camera->Close();
    if (err != VmbErrorSuccess)
    {
        std::cerr << "Could not disconnect from camera..." << std::endl;
    }

    this->cameraOpen = false;
    return this->cameraOpen;
}

bool AlliedVisionAlvium::getSingleFrame(cv::Mat& buffer, uint64_t& cameraFrameID, uint64_t& cameraTimestamp, uint32_t timeoutMs)
{
    VmbCPP::FramePtr frame;
    VmbError_t err;
    int openCvType;
    VmbPixelFormatType format;
    VmbFrameStatusType status;
    uint32_t height;
    uint32_t width;
    uint32_t bufferSize;
    cv::Mat image;
    uint8_t* data;
    VmbUint64_t timestamp;
    VmbUint64_t frameID;
    /* These are used for unpacking images if need be */
    VmbImage sourceImage;
    VmbImage destinationImage;
    bool requiresUnpacking;

    err = camera->AcquireSingleImage(frame, timeoutMs);
    if (VmbErrorSuccess != err)
    {
        std::cerr << "Could not get single frame: " << err << std::endl;
        return false;
    }

    err = frame->GetReceiveStatus(status);

    if (VmbErrorSuccess != err)
    {
        std::cerr << "Could not get frame status" << std::endl;
        return false;
    }
    else if (VmbFrameStatusComplete != status)
    {
        switch (status)
        {
        case VmbFrameStatusIncomplete:
        {
            std::cerr << "Frame incomplete. Try a slower frame rate" << std::endl;
            return false;
        }
        case VmbFrameStatusTooSmall:
        {
            std::cerr << "Frame too small..." << std::endl;
            return false;
        }
        case VmbFrameStatusInvalid:
        {
            std::cerr << "Frame invalid..." << std::endl;
            return false;
        }
        }
    }

    frame->GetPixelFormat(format);
    frame->GetHeight(height);
    frame->GetWidth(width);
    frame->GetTimestamp(timestamp);
    cameraTimestamp = timestamp;
    frame->GetFrameID(frameID);
    cameraFrameID = frameID;
    frame->GetBufferSize(bufferSize);
    frame->GetBuffer(data);
    switch (format)
    {
    case VmbPixelFormatMono8:
    {
        openCvType = CV_8UC1;
        image = cv::Mat(
            height,
            width,
            openCvType,
            data);
        break;
    }
    case VmbPixelFormatMono10:
    {
        openCvType = CV_16UC1;
        image = cv::Mat(
            height,
            width,
            openCvType,
            data);
        break;
    }
    case VmbPixelFormatMono12:
    {
        openCvType = CV_16UC1;
        image = cv::Mat(
            height,
            width,
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
        destinationImage.Data = malloc(width * height * 2);
        if (nullptr == destinationImage.Data)
        {
            std::cerr << "Could not create destination buffer for unpacking" << std::endl;
            return false;
        }

        VmbError_t error = VmbSetImageInfoFromPixelFormat(
            VmbPixelFormatMono12p,
            width,
            height,
            &sourceImage);
        if (VmbErrorSuccess != error)
        {
            std::cerr << "Could not create source image info for unpacking: " << error << std::endl;
            free(destinationImage.Data);
            return false;
        }
        error = VmbSetImageInfoFromInputParameters(
            VmbPixelFormatMono12,
            width,
            height,
            VmbPixelLayoutMono,
            16,
            &destinationImage);
        if (error != VmbErrorSuccess)
        {
            std::cerr << "Could not create destination image info for unpacking: " << error << std::endl;
            return false;
        }
        error = VmbImageTransform(
            &sourceImage,
            &destinationImage,
            NULL,
            0);
        if (error != VmbErrorSuccess)
        {
            std::cerr << "Could not unpack image: " << error << std::endl;
            return false;
        }
        image = cv::Mat(
            height,
            width,
            openCvType,
            destinationImage.Data);
        free(destinationImage.Data);
        break;
    }
    default:
    {
        std::cerr << "Camera frame format not supported... " << format << std::endl;
        free(destinationImage.Data);
        return false;
    }
    }

    /* returns the frame buffer back to the queue */
    this->camera->QueueFrame(frame);
    buffer = image.clone();

    return true;
}

bool AlliedVisionAlvium::startAcquisition(
    int bufferCount,
    std::function<void(AlliedVisionAlviumFrameData&, void*)> newFrameCallback,
    void* arg)
{
    VmbErrorType err;
    err = camera->StartContinuousImageAcquisition(
        bufferCount,
        VmbCPP::IFrameObserverPtr(new FrameObserver(camera, newFrameCallback, arg)));
    if (VmbErrorSuccess != err)
    {
        std::cerr << "Unable to start image Aqcuisition... " << err << std::endl;

        return false;
    }

    return true;
}

bool AlliedVisionAlvium::stopAcquisition(void)
{
    VmbErrorType err;
    err = camera->StopContinuousImageAcquisition();
    if (VmbErrorSuccess != err)
    {
        return false;
    }

    return true;
}

bool AlliedVisionAlvium::isCameraOpen(void)
{
    return this->cameraOpen;
}
std::string AlliedVisionAlvium::getName()
{
    VmbError_t err;
    std::string cameraName;
    if (false == this->cameraOpen)
    {
        std::cerr << "Could not get camera name. No camera is open..." << err << std::endl;
        return "";
    }

    err = this->camera->GetName(cameraName);
    if (err != VmbErrorSuccess)
    {
        std::cerr << "Unable to get camera name..." << err << std::endl;
        return "";
    }

    return cameraName;
}
bool AlliedVisionAlvium::setFeature(
    std::string featureName,
    std::string featureValue)
{
    VmbCPP::FeaturePtr feature;
    VmbError_t error;
    VmbFeatureDataType dataType;

    error = this->camera->GetFeatureByName(featureName.c_str(), feature);
    if (VmbErrorSuccess != error)
    {
        std::cerr << "Could not get feature " << featureName << std::endl;
        return false;
    }

    error = feature->GetDataType(dataType);
    if (VmbErrorSuccess != error)
    {
        std::cerr << "Could not get feature " << featureName << " datatype" << std::endl;
        return false;
    }

    if (VmbFeatureDataInt == dataType)
    {
        error = feature->SetValue(std::stoi(featureValue));
    }
    else if (VmbFeatureDataFloat == dataType)
    {
        error = feature->SetValue(std::stof(featureValue));
    }
    else if (VmbFeatureDataEnum == dataType)
    {
        error = feature->SetValue(featureValue.c_str());
    }
    else if (VmbFeatureDataString == dataType)
    {
        error = feature->SetValue(featureValue.c_str());
    }
    else if (VmbFeatureDataBool == dataType)
    {
        if (featureValue == "false")
        {
            error = feature->SetValue(false);
        }
        else if (featureValue == "true")
        {
            error = feature->SetValue(true);
        }
    }
    else if (VmbFeatureDataCommand == dataType)
    {
        feature->RunCommand();
    }
    else
    {
        std::cerr << "Unknown feature datatype: " << dataType << " for " << featureName << std::endl;
        return false;
    }

    if (VmbErrorSuccess != error)
    {
        std::cerr << "Could not set feature " << featureName << ": " << error << std::endl;
        return false;
    }

    return true;
}

bool AlliedVisionAlvium::getFeature(
    std::string featureName,
    std::string& featureValue)
{
    VmbCPP::FeaturePtr feature;
    VmbError_t error;
    VmbFeatureDataType dataType;

    error = this->camera->GetFeatureByName(featureName.c_str(), feature);
    if (VmbErrorSuccess != error)
    {
        std::cerr << "Could not get feature " << featureName << ": " << error << std::endl;
        return false;
    }
    error = feature->GetDataType(dataType);
    if (VmbErrorSuccess != error)
    {
        std::cerr << "Could not get feature " << featureName << " datatype" << std::endl;
        return false;
    }

    if (VmbFeatureDataInt == dataType)
    {
        VmbInt64_t data;
        error = feature->GetValue(data);
        if (VmbErrorSuccess != error)
        {
            std::cerr << "Could not get feature value" << featureName << ": " << error << std::endl;
            return false;
        }
        featureValue = std::to_string(data);
    }
    else if (VmbFeatureDataFloat == dataType)
    {
        double data;
        error = feature->GetValue(data);
        if (VmbErrorSuccess != error)
        {
            std::cerr << "Could not get feature value" << featureName << ": " << error << std::endl;
            return false;
        }
        featureValue = std::to_string(data);
    }
    else if (VmbFeatureDataString == dataType)
    {
        std::string data;
        error = feature->GetValue(data);
        if (VmbErrorSuccess != error)
        {
            std::cerr << "Could not get feature value" << featureName << ": " << error << std::endl;
            return false;
        }
        featureValue = data;
    }
    else if (VmbFeatureDataBool == dataType)
    {
        bool data;
        error = feature->GetValue(data);
        if (VmbErrorSuccess != error)
        {
            std::cerr << "Could not get feature value" << featureName << ": " << error << std::endl;
            return false;
        }
        featureValue = std::to_string(data);
    }
    else if (VmbFeatureDataEnum == dataType)
    {
        std::string data;
        error = feature->GetValue(data);
        if (VmbErrorSuccess != error)
        {
            std::cerr << "Could not get feature value" << featureName << ": " << error << std::endl;
            return false;
        }
        featureValue = data;
    }
    else
    {
        std::cerr << "Unknown feature datatype: " << dataType << " for " << featureName << std::endl;
        return false;
    }

    return true;
}

bool AlliedVisionAlvium::activateEvent(
    std::string eventName,
    std::string eventObserverName,
    std::function<void(
        std::string,
        int64_t,
        time_t,
        time_t,
        void*)>
    eventCallback,
    void* arg)
{
    VmbCPP::FeaturePtr pFeature;
    VmbError_t error;

    /* First activate the event */

    // EventSelector is used to specify the particular Event to control
    VmbCPP::FeaturePtr feature;
    error = this->camera->GetFeatureByName("EventSelector", feature);
    if (error != VmbErrorSuccess)
    {
        return false;
    }

    error = feature->SetValue(eventName.c_str());
    if (error != VmbErrorSuccess)
    {
        return false;
    }

    // EventNotification is used to enable/disable the notification of the event specified by EventSelector.
    error = this->camera->GetFeatureByName("EventNotification", feature);
    if (error != VmbErrorSuccess)
    {
        return false;
    }
    else
    {
        error = feature->SetValue("On");
    }

    /* Register the event */
    std::cout << "Registering observer for " << eventObserverName << " feature." << std::endl;
    error = this->camera->GetFeatureByName(eventObserverName.c_str(), pFeature);

    if (error == VmbErrorSuccess)
    {
        // register a callback function to be notified that the event happened
        error = pFeature->RegisterObserver(
            VmbCPP::IFeatureObserverPtr(new EventObserver(eventCallback, arg)));

        if (error != VmbErrorSuccess)
        {
            std::cout << "Could not register observer. Error code: " << error << "\n";
            return false;
        }
    }
    else
    {
        std::cout << "Could not register observer. Could not get feature: " << eventObserverName << std::endl;
        return false;
    }

    return true;
}

bool AlliedVisionAlvium::runCommand(
    std::string command)
{
    VmbCPP::FeaturePtr feature;
    VmbError_t error;

    error = this->camera->GetFeatureByName(command.c_str(), feature);
    if (VmbErrorSuccess != error)
    {
        std::cerr << "Could not get feature " << command << ": " << error << std::endl;
        return false;
    }

    error = feature->RunCommand();
    if (VmbErrorSuccess != error)
    {
        std::cerr << "Could not run command " << command << ": " << error << std::endl;
        return false;
    }

    bool ret;
    while (true)
    {
        error = feature->IsCommandDone(ret);

        if (VmbErrorSuccess != error)
        {
            std::cerr << "Could not wait for command " + command << ": " << error << std::endl;
            return false;
        }
        else if (true == ret)
        {
            break;
        }
    }

    return true;
}

#endif // ALLIEDVISIONALVIUM_H_
