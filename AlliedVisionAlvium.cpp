#include "AlliedVisionAlvium.hpp"
#include <iostream>
#include <VmbCPP/Frame.h>

VmbError_t VMB_CALL ChunkCallback(VmbHandle_t featureAccessHandle, void *userContext)
{
    AlliedVisionAlviumFrameData *frameData;
    frameData = (AlliedVisionAlviumFrameData *)userContext;

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

/**
 * \brief Handles a received frame from the camera.
 *
 * This function processes an incoming frame, extracting metadata and image data.
 * It retrieves the frame status, dimensions, pixel format, and chunk data such
 * as exposure time and gain. Depending on the pixel format, it may unpack the
 * image data into a cv::Mat for further processing. Once processed, the frame
 * is returned to the queue, and a callback is triggered with the frame data.
 *
 * \param frame A smart pointer to the received frame.
 */

void FrameObserver::FrameReceived(const VmbCPP::FramePtr frame)
{
    timespec ts;
    auto now = clock_gettime(CLOCK_REALTIME, &ts);
    AlliedVisionAlviumFrameData frameData;

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
    frameData.cameraFrameStartTimestamp = cameraTimestamp;
    frameData.cameraFrameId = frameID;

    VmbFrame_t *frame2GetChunk = nullptr;
    err = frame->GetFrameStruct(frame2GetChunk);
    if (err != VmbErrorSuccess)
    {
        std::cerr << "Could not get frame struct:" << err << std::endl;
    }

    err = VmbChunkDataAccess(frame2GetChunk, ChunkCallback, &frameData);
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

/**
 * \brief Called when a feature has changed
 *
 * This method is called by the underlying Vimba C++ API when a feature has
 * changed. The feature that has changed is passed as an argument to the method.
 *
 * When this method is called, it will call the callback function set with the
 * eventCallback argument to the EventObserver constructor. The callback function
 * will receive the name of the feature that has changed, the new value of the
 * feature, the current time in seconds and nanoseconds, and the argument set
 * with the argument argument to the EventObserver constructor.
 *
 * The callback function may be nullptr if no callback should be called when an
 * event occurs.
 */
void EventObserver::FeatureChanged(const VmbCPP::FeaturePtr &feature)
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

/**
 * \brief Constructor for the AlliedVisionAlvium class
 *
 * Initializes the AlliedVisionAlvium object. This constructor does not
 * perform any operations, it is primarily used to set up the object
 * instance for further configuration and operations.
 */

AlliedVisionAlvium::AlliedVisionAlvium()
{
}

/**
 * \brief Destructor for the AlliedVisionAlvium class
 *
 * Safely disconnects from the camera if it is open when the
 * AlliedVisionAlvium object is destroyed. This ensures that
 * resources are properly released and any exceptions during
 * disconnection are caught and logged.
 */

AlliedVisionAlvium::~AlliedVisionAlvium()
{
    try
    {
        if (cameraOpen)
        {
            this->disconnect();
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what() << '\n';
    }
}

/**
 * \brief Retrieve the device ID of a camera given its user ID.
 *
 * This method takes a camera user ID as input and attempts to find the
 * matching device ID from the list of cameras available. It goes through
 * all the cameras, opens each one, checks if the user ID matches and if
 * so, retrieves the device ID and returns it. If the camera user ID does
 * not match, the camera is closed and the search continues. If the camera
 * user ID is not found, false is returned.
 *
 * \param[in] cameraUserId The user ID of the camera to search for.
 * \param[out] deviceID The device ID of the camera which matches the
 * given user ID.
 * \return True if the camera user ID is found and the device ID is
 * retrieved, false otherwise.
 */
bool AlliedVisionAlvium::getCameraUserIdFromDeviceIdList(
    std::string cameraUserId,
    std::string &deviceID)
{
    VmbCPP::CameraPtrVector cameras;

    VmbCPP::VmbSystem &vimbax =
        VmbCPP::VmbSystem::GetInstance();

    VmbErrorType err;
    /* Start the API, get and open cameras */
    err = vimbax.Startup();

    if (err != VmbErrorSuccess && err != VmbErrorAlready)
    {
        std::cerr << "AlliedVisionAlvium::getCameraUserIdFromDeviceIdList: Unable to start up vimbax " << std::endl;
        return false;
    }
    else if (VmbErrorSuccess != vimbax.GetCameras(cameras))
    {
        std::cerr << "AlliedVisionAlvium::getCameraUserIdFromDeviceIdList: Unable to detect any cameras" << std::endl;
        return false;
    }

    for (int i = 0; i < cameras.size(); i++)
    {
        std::string userId;
        VmbCPP::FeaturePtr feature;
        VmbErrorType err = cameras[i]->Open(VmbAccessModeFull);

        if (VmbErrorSuccess != err)
        {
            std::cerr << "AlliedVisionAlvium::getCameraUserIdFromDeviceIdList: Unable to open camera: " << err << std::endl;
            cameras[i]->Close();
            continue;
        }

        err = cameras[i]->GetFeatureByName("DeviceUserID", feature);
        if (VmbErrorSuccess != err)
        {
            std::cerr << "AlliedVisionAlvium::getCameraUserIdFromDeviceIdList: Could not get feature: " << err << std::endl;
            cameras[i]->Close();
            continue;
        }

        err = feature->GetValue(userId);
        if (VmbErrorSuccess != feature->GetValue(userId))
        {
            std::cerr << "AlliedVisionAlvium::getCameraUserIdFromDeviceIdList: Unable to get userdeviceid" << err << std::endl;
            cameras[i]->Close();
            continue;
        }

        if (userId != cameraUserId)
        {
            cameras[i]->Close();
            continue;
        }

        err = cameras[i]->GetID(deviceID);
        if (VmbErrorSuccess != err)
        {
            std::cerr << "AlliedVisionAlvium::getCameraUserIdFromDeviceIdList: Unable to get device Id: " << err << std::endl;
            cameras[i]->Close();
            continue;
        }
        else
        {
            cameras[i]->Close();
            return true;
        }
    }

    /* if we are up to here we have not found the right camera name */
    return false;
}

/**
 * \brief Connects to the first available camera and initializes it.
 *
 * This function attempts to start the Vimba API and retrieve a list of connected cameras.
 * It iterates through the available cameras to obtain their names and IDs. Once a camera
 * is successfully opened using its ID, it sets specific features such as enabling chunk
 * data for exposure time and gain. If no camera can be connected or any operation fails,
 * the function logs an error and returns false. If a connection is established, it logs
 * the connected camera's ID and returns true.
 *
 * \return True if a camera is successfully connected and initialized, false otherwise.
 */

bool AlliedVisionAlvium::connect()
{
    VmbErrorType err;
    VmbCPP::CameraPtrVector cameras;
    std::string cameraID;
    VmbCPP::VmbSystem &vimbax =
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
    else
    {
        for (int i = 0; i < cameras.size(); i++)
        {
            std::string name;
            err = cameras[i]->GetName(name);
            if (err != VmbErrorSuccess)
            {
                std::cerr << "Unable to get camera name : " << err << std::endl;
                this->cameraOpen = false;
                return false;
            }
        }
    }

    for (int i = 0; i < cameras.size(); i++)
    {
        err = cameras[i]->GetID(cameraID);
        if (err != VmbErrorSuccess)
        {
            std::cerr << "Unable to get camera name : " << err << std::endl;
            this->cameraOpen = false;
            return false;
        }

        err = vimbax.OpenCameraByID(
            cameraID,
            VmbAccessModeFull,
            this->camera);
        if (err != VmbErrorSuccess)
        {
            std::cerr << "Unable to connect to camera " << cameraID << ": " << err << std::endl;
            this->cameraOpen = false;
        }
        else
        {
            this->cameraOpen = true;
            break;
        }
    }

    if (false == this->cameraOpen)
    {
        std::cerr << "Unable to connect to any camera" << err << std::endl;
        return false;
    }

    if (false == this->enableChunk())
    {
        std::cerr << "Unable to enable chunk data" << std::endl;
        return false;
    }

    std::cout << "Connected to " << cameraID << std::endl;
    return this->cameraOpen;
}

/**
 * \brief Connect to a camera by user ID.
 *
 * This function takes a user ID as input and attempts to find the
 * matching device ID from the list of cameras available. It goes through
 * all the cameras, opens each one, checks if the user ID matches and if
 * so, retrieves the device ID and returns true. If the camera user ID does
 * not match, the camera is closed and the search continues. If the camera
 * user ID is not found, false is returned.
 *
 * \param[in] userId The user ID of the camera to search for.
 * \return True if the camera user ID is found and the device ID is
 * retrieved, false otherwise.
 */
bool AlliedVisionAlvium::connectByUserId(std::string userId)
{
    VmbErrorType err;
    VmbCPP::VmbSystem &vimbax =
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

    std::string deviceID;
    if (false == this->getCameraUserIdFromDeviceIdList(userId, deviceID))
    {
        std::cerr << userId << " not detected..." << std::endl;
    }
    err = vimbax.OpenCameraByID(
        deviceID,
        VmbAccessModeFull,
        this->camera);
    if (err != VmbErrorSuccess)
    {
        std::cerr << "Unable to connect to camera " << deviceID << ": " << err << std::endl;
        this->cameraOpen = false;
        return false;
    }

    if (false == this->enableChunk())
    {
        std::cerr << "Unable to enable chunk data" << std::endl;
        return false;
    }

    std::cout << "Connected to " << userId << "/" << deviceID << std::endl;

    this->cameraOpen = true;
    return this->cameraOpen;
}

/**
 * \brief Get a list of user IDs for all cameras that are currently plugged in.
 *
 * This function starts up the Vimba API and retrieves a list of all cameras
 * that are currently plugged in. It then iterates through each camera, opens
 * it, gets the device user ID and closes it. If any of the steps fail, the
 * camera is skipped and the function continues with the next camera. If no
 * cameras are plugged in, an empty vector is returned.
 *
 * \return A vector of strings containing the user IDs of all cameras that are
 * plugged in.
 */
std::vector<std::string> AlliedVisionAlvium::getUserIds()
{
    VmbCPP::CameraPtrVector cameras;
    std::vector<std::string> cameraUserIds;

    VmbCPP::VmbSystem &vimbax =
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
    }
    else if (VmbErrorSuccess != vimbax.GetCameras(cameras))
    {
        std::cerr << "AlliedVisionAlvium::getCameraList: Unable to detect any cameras" << std::endl;
    }
    for (int i = 0; i < cameras.size(); i++)
    {
        std::string userId;
        VmbCPP::FeaturePtr feature;
        if (VmbErrorSuccess != cameras[i]->Open(VmbAccessModeFull))
        {
            std::cerr << "AlliedVisionAlvium::getUserIds: Unable to open camera" << std::endl;
            camera->Close();
            continue;
        }

        err = cameras[i]->GetFeatureByName("DeviceUserID", feature);
        if (VmbErrorSuccess != err)
        {
            std::cerr << "AlliedVisionAlvium::getCameraUserIdFromDeviceIdList: Could not get feature" << err << std::endl;
            cameras[i]->Close();
            continue;
        }

        err = feature->GetValue(userId);
        if (VmbErrorSuccess != err)
        {
            std::cerr << "AlliedVisionAlvium::getCameraUserIdFromDeviceIdList: Unable to get userdeviceid: " << err << std::endl;
            cameras[i]->Close();
            continue;
        }

        cameraUserIds.push_back(userId);
        cameras[i]->Close();
    }
    return cameraUserIds;
}

/**
 * \brief Disconnects from the currently connected camera.
 *
 * This function stops any ongoing image acquisition and attempts to close
 * the connection to the camera. It ensures that the camera is properly
 * disconnected and updates the cameraOpen flag accordingly. If the camera
 * cannot be disconnected successfully, an error message is logged.
 *
 * \return False if the camera is successfully disconnected, true otherwise.
 */
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

/**
 * \brief Grabs a single frame from the camera and stores it in a buffer.
 *
 * This function waits for a single frame from the camera and stores it in the
 * buffer provided. It includes the timestamp when the frame was received and
 * the frame ID. The buffer is also populated with the exposure time and gain
 * used for the frame. The image is stored in the buffer as an OpenCV Mat. The
 * function will wait for up to timeoutMs milliseconds for a frame to be
 * received. If no frame is received in that time, false is returned.
 *
 * \param buffer The buffer to store the frame in.
 * \param timeoutMs The maximum time in milliseconds to wait for a frame.
 *
 * \return True if a frame was successfully received and stored in the buffer,
 * false otherwise.
 */
bool AlliedVisionAlvium::getSingleFrame(
    AlliedVisionAlviumFrameData &buffer,
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

    timespec ts;
    auto now = clock_gettime(CLOCK_REALTIME, &ts);

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
        this->camera->QueueFrame(frame);

        return false;
    }
    else if (VmbFrameStatusComplete != status)
    {
        switch (status)
        {
        case VmbFrameStatusIncomplete:
        {
            std::cerr << "Frame incomplete. Try a slower frame rate" << std::endl;
            this->camera->QueueFrame(frame);

            return false;
        }
        case VmbFrameStatusTooSmall:
        {
            std::cerr << "Frame too small..." << std::endl;
            this->camera->QueueFrame(frame);

            return false;
        }
        case VmbFrameStatusInvalid:
        {
            std::cerr << "Frame invalid..." << std::endl;
            this->camera->QueueFrame(frame);

            return false;
        }
        }
    }

    // Access the Chunk data of the incoming frame. Chunk data accesible inside lambda function
    err = frame->AccessChunkData(
        [this, &buffer](VmbCPP::ChunkFeatureContainerPtr &chunkFeatures) -> VmbErrorType
        {
            VmbCPP::FeaturePtr feat;
            VmbErrorType err;

            std::string exposure = "";
            std::string gain = "";
            buffer.exposureTimeUs = 0;
            buffer.gainDb = 0;

            /* test if chunkFeatures is valid*/
            // Check if the chunkFeatures smart pointer is valid (not null).
            if (chunkFeatures->GetHandle() == nullptr)
            {
                std::cerr << "Could not access chunk features: the pointer is null." << std::endl;
                return VmbErrorNotFound; // Return an appropriate error code.
            }

            /* Get all the availabe features in the ChunkData */
            err = chunkFeatures->GetFeatureByName("ExposureTime", feat);
            if (err != VmbErrorSuccess)
            {
                std::cerr << "Could not get ExposureTime feature from ChunkData: " << err << std::endl;
                return VmbErrorCustom;
            }

            err = feat->GetValue(buffer.exposureTimeUs);
            if (err != VmbErrorSuccess)
            {
                std::cerr << "Could not get ExposureTime value from ChunkData: " << err << std::endl;
                return VmbErrorCustom;
            }

            /* Get all the availabe features in the ChunkData */
            err = chunkFeatures->GetFeatureByName("Gain", feat);
            if (err != VmbErrorSuccess)
            {
                std::cerr << "Could not get Gain feature from ChunkData: " << err << std::endl;
                return VmbErrorCustom;
            }

            err = feat->GetValue(buffer.gainDb);
            if (err != VmbErrorSuccess)
            {
                std::cerr << "Could not get Gain value from ChunkData: " << err << std::endl;
                return VmbErrorCustom;
            }

            return VmbErrorSuccess;
        });

    if (err != VmbErrorSuccess)
    {
        std::cerr << "Could not access frame ChunkData:" << err << std::endl;

        std::string exposure;
        if (true == this->getFeature("ExposureTime", exposure))
        {
            buffer.exposureTimeUs = std::stod(exposure);
        }
        else
        {
            buffer.exposureTimeUs = 0;
        }

        std::string gain;
        if (true == this->getFeature("Gain", gain))
        {
            buffer.gainDb = std::stod(gain);
        }
        else
        {
            buffer.gainDb = 0;
        }
    }

    VmbUint64_t cameraTimestamp;

    frame->GetPixelFormat(format);
    frame->GetBufferSize(bufferSize);
    frame->GetBuffer(data);
    frame->GetTimestamp(cameraTimestamp);
    frame->GetFrameID(frameID);
    frame->GetHeight(buffer.height);
    frame->GetWidth(buffer.width);
    frame->GetOffsetX(buffer.offsetX);
    frame->GetOffsetY(buffer.offsetY);
    buffer.cameraFrameStartTimestamp = cameraTimestamp;
    buffer.cameraFrameId = frameID;
    buffer.systemImageReceivedTimestamp =
        ts.tv_sec * 10000000000 + ts.tv_nsec;

    cv::Mat tempImage;
    switch (format)
    {
    case VmbPixelFormatMono8:
    {
        openCvType = CV_8UC1;
        tempImage = cv::Mat(
            buffer.height,
            buffer.width,
            openCvType,
            data);
        buffer.image = tempImage.clone();

        break;
    }
    case VmbPixelFormatMono10:
    {
        openCvType = CV_16UC1;
        tempImage = cv::Mat(
            buffer.height,
            buffer.width,
            openCvType,
            data);
        buffer.image = tempImage.clone();

        break;
    }
    case VmbPixelFormatMono12:
    {
        openCvType = CV_16UC1;
        tempImage = cv::Mat(
            buffer.height,
            buffer.width,
            openCvType,
            data);
        buffer.image = tempImage.clone();

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
        destinationImage.Data = malloc(buffer.width * buffer.height * 2);
        if (nullptr == destinationImage.Data)
        {
            std::cerr << "Could not create destination buffer for unpacking" << std::endl;
            return false;
        }

        VmbError_t error = VmbSetImageInfoFromPixelFormat(
            VmbPixelFormatMono12p,
            buffer.width,
            buffer.height,
            &sourceImage);
        if (VmbErrorSuccess != error)
        {
            std::cerr << "Could not create source image info for unpacking: " << error << std::endl;
            free(destinationImage.Data);
            return false;
        }
        error = VmbSetImageInfoFromInputParameters(
            VmbPixelFormatMono12,
            buffer.width,
            buffer.height,
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
        tempImage = cv::Mat(
            buffer.height,
            buffer.width,
            openCvType,
            destinationImage.Data);
        buffer.image = tempImage.clone();
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

    return true;
}

bool AlliedVisionAlvium::enableExternalTrigger(AlliedVisionAlvium::Line line)
{
    /* Configure the camera for external triggering */
    std::string linestr;
    switch (line)
    {
    case Line::LINE0:
        linestr = "Line0";
        break;
    case Line::LINE1:
        linestr = "Line1";
        break;
    case Line::LINE2:
        linestr = "Line2";
        break;
    case Line::LINE3:
        linestr = "Line3";
        break;
    default:
        std::cerr << "Unknown line: " << line << std::endl;
        return false;
    }

    if (false == this->setFeature("AcquisitionFrameRateEnable", "false"))
    {
        std::cerr << "Could not set AcquisitionFrameRateEnable as false" << std::endl;
        return false;
    }
    else if (false == this->setFeature("LineSelector", linestr))
    {
        std::cerr << "Could not set LineSelector as " << linestr << std::endl;
        return false;
    }
    else if (false == this->setFeature("LineMode", "Input"))
    {
        std::cerr << "Could not set LineMode as Input" << std::endl;
        return false;
    }
    else if (false == this->setFeature("LineDebounceMode", "Off"))
    {
        std::cerr << "Could not set LineDebounceMode as Off" << std::endl;
        return false;
    }
    else if (false == this->setFeature("TriggerSelector", "FrameStart"))
    {
        std::cerr << "Could not set TriggerSelector as FrameStart" << std::endl;
        return false;
    }
    else if (false == this->setFeature("TriggerSource", linestr))
    {
        std::cerr << "Could not set TriggerSource as " << linestr << std::endl;
        return false;
    }
    else if (false == this->setFeature("TriggerMode", "On"))
    {
        std::cerr << "Could not set TriggerMode as On" << std::endl;
        return false;
    }

    return true;
}

bool AlliedVisionAlvium::disableExternalTrigger(void)
{
    if (false == this->setFeature("AcquisitionFrameRateEnable", "false"))
    {
        std::cerr << "Could not set AcquisitionFrameRateEnable as false" << std::endl;
        return false;
    }
    else if (false == this->setFeature("TriggerMode", "Off"))
    {
        std::cerr << "Could not set TriggerMode as Off" << std::endl;
        return false;
    }

    return true;
}

/**
 * \brief Starts continuous image acquisition.
 *
 * This function initiates continuous image acquisition from the camera.
 * It sets up the frame observer using the provided callback for processing
 * new frames and an optional argument that can be used within the callback.
 *
 * \param bufferCount The number of image buffers to be used for acquisition.
 * \param newFrameCallback A callback function that is called with each new frame.
 * \param arg An optional argument passed to the callback function.
 * \return True if the acquisition starts successfully, false otherwise.
 */

bool AlliedVisionAlvium::startAcquisition(
    int bufferCount,
    std::function<void(AlliedVisionAlviumFrameData &, void *)> newFrameCallback,
    void *arg)
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

/**
 * \brief Stops continuous image acquisition.
 *
 * This function stops continuous image acquisition from the camera and
 * releases all queued image buffers.
 *
 * \return True if the acquisition stops successfully, false otherwise.
 */
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

/**
 * \brief Checks if the camera is open.
 *
 * This function returns the status of the camera to indicate whether it
 * is currently open and available for operations.
 *
 * \return True if the camera is open, false otherwise.
 */
bool AlliedVisionAlvium::isCameraOpen(void)
{
    return this->cameraOpen;
}

/**
 * \brief Gets the name of the camera.
 *
 * This function returns the name of the currently open camera.
 *
 * \return The name of the camera if successful, an empty string otherwise.
 */
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

/**
 * \brief Gets the user ID of the currently open camera.
 *
 * This function attempts to retrieve the user ID of the camera that is currently
 * open. If the camera is not open, an error message is logged and an empty string
 * is returned. If the user ID cannot be retrieved, an error message is logged and
 * an empty string is returned.
 *
 * \return The user ID of the camera if successful, an empty string otherwise.
 */
std::string AlliedVisionAlvium::getUserId()
{
    VmbError_t err;
    std::string userId;
    if (false == this->cameraOpen)
    {
        std::cerr << "Could not get camera name. No camera is open..." << err << std::endl;
        return "";
    }

    if (false == this->getFeature("DeviceUserID", userId))
    {
        std::cerr << "Unable to get userId..." << std::endl;
        return "";
    }

    return userId;
}

/**
 * \brief Sets the value of a feature.
 *
 * This function takes a feature name and a new value as arguments and sets the
 * value of the feature. The function attempts to retrieve the feature by name
 * and then sets the value with the appropriate data type. If the feature does
 * not exist, an error is logged and false is returned. If the feature cannot
 * be set, an error is logged and false is returned.
 *
 * \param featureName The name of the feature to set.
 * \param featureValue The new value of the feature.
 * \return True if the feature is set successfully, false otherwise.
 */
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

bool AlliedVisionAlvium::enableChunk()
{
    /* Enable the chunk data for different things so we can embedd them in the image data*/
    if (false == this->setFeature("ChunkModeActive", "false"))
    {
        std::cerr << "Unable to set ChunkModeActive to false" << std::endl;
        return false;
    }

    if (false == this->setFeature("ChunkSelector", "ExposureTime"))
    {
        std::cerr << "Unable to set ChunkSelector to ExposureTime" << std::endl;
    }
    else if (false == this->setFeature("ChunkEnable", "true"))
    {
        std::cerr << "Unable to set ChunkEnable ExposureTime to true" << std::endl;
    }
    else
    {
        std::cerr << "Set ChunkEnable ExposureTime to true" << std::endl;
    }

    if (false == this->setFeature("ChunkSelector", "Gain"))
    {
        std::cerr << "Unable to set ChunkSelector to Gain" << std::endl;
    }
    else if (false == this->setFeature("ChunkEnable", "true"))
    {
        std::cerr << "Unable to set ChunkEnable Gain to true" << std::endl;
    }

    if (false == this->setFeature("ChunkModeActive", "true"))
    {
        std::cerr << "Unable to set ChunkModeActive to true" << std::endl;
        return false;
    }

    return true;
}
/**
 * \brief Retrieves the value of a camera feature.
 *
 * This function attempts to retrieve the value of a specified camera feature
 * by its name. It determines the feature's data type and fetches the value
 * accordingly. Supported data types include integer, float, string, boolean,
 * and enum. If the feature cannot be retrieved or the data type is unknown,
 * an error is logged and the function returns false.
 *
 * \param[in] featureName The name of the feature to retrieve.
 * \param[out] featureValue The retrieved value of the feature as a string.
 * \return True if the feature value is retrieved successfully, false otherwise.
 */
bool AlliedVisionAlvium::getFeature(
    std::string featureName,
    std::string &featureValue)
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
            std::cerr << "Could not get feature value " << featureName << ": " << error << std::endl;
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
            std::cerr << "Could not get feature value " << featureName << ": " << error << std::endl;
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

/**
 * \brief Activates an event.
 *
 * This function takes an event name and an event observer name as arguments and
 * sets the value of the event selector feature to the specified event name.
 * It also sets the value of the event notification feature to "On".
 * Finally, it registers a callback function to be notified that the event
 * happened.
 *
 * \param eventName The name of the event to activate.
 * \param eventObserverName The name of the feature to observe for the event.
 * \param eventCallback The callback function to execute when the event happens.
 * \param arg The argument to pass to the callback function.
 * \return True if the event is successfully activated, false otherwise.
 */
bool AlliedVisionAlvium::activateEvent(
    std::string eventName,
    std::string eventObserverName,
    std::function<void(
        std::string,
        int64_t,
        time_t,
        time_t,
        void *)>
        eventCallback,
    void *arg)
{
    VmbCPP::FeaturePtr pFeature;
    VmbError_t error;

    /* First activate the event */

    // EventSelector is used to specify the particular Event to control
    VmbCPP::FeaturePtr feature;
    error = this->camera->GetFeatureByName("EventSelector", feature);
    if (error != VmbErrorSuccess)
    {
        std::cerr << "Could not get feature EventSelector" << error << std::endl;
        return false;
    }

    error = feature->SetValue(eventName.c_str());
    if (error != VmbErrorSuccess)
    {
        std::cerr << "Could not set feature EventSelector as " << eventName << ": " << error << std::endl;
        return false;
    }

    // EventNotification is used to enable/disable the notification of the event specified by EventSelector.
    error = this->camera->GetFeatureByName("EventNotification", feature);
    if (error != VmbErrorSuccess)
    {
        std::cerr << "Could not get feature EventNotification" << error << std::endl;
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

/**
 * \brief Run a command feature on the camera.
 *
 * This function takes a command feature name, attempts to retrieve the feature
 * from the camera, runs the feature, and waits until the command is done.
 *
 * @param command The name of the command feature to run.
 * @return True if the command runs successfully, false otherwise.
 */
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

VmbErrorType AlliedVisionAlvium::getFeatureStatic(
    VmbCPP::FeaturePtr feat,
    std::string &val)
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

void AlliedVisionAlvium::CameraListChanged(
    VmbCPP::CameraPtr pCam,
    VmbCPP::UpdateTriggerType reason)
{
    if (reason == VmbCPP::UpdateTriggerPluggedIn)
    {
        VmbErrorType err;
        VmbCPP::CameraPtrVector cameras;
        std::string cameraName;
        std::string cameraId;

        err = pCam->GetName(cameraName);
        if (err != VmbErrorSuccess)
        {
            std::cerr << "Could not get camera name..." << std::endl;
        }

        err = pCam->GetID(cameraId);
        if (err != VmbErrorSuccess)
        {
            std::cerr << "Could not get camera id..." << std::endl;
        }
        std::cout << "Dectected camera plug in: " << cameraName << ":" << cameraId << std::endl;
    }
    else if (reason == VmbCPP::UpdateTriggerPluggedOut)
    {
        VmbErrorType err;
        VmbCPP::CameraPtrVector cameras;
        std::string cameraName;
        std::string cameraId;
        err = pCam->GetName(cameraName);
        if (err != VmbErrorSuccess)
        {
            std::cerr << "Could not get camera name..." << std::endl;
        }

        err = pCam->GetID(cameraId);
        if (err != VmbErrorSuccess)
        {
            std::cerr << "Could not get camera id..." << std::endl;
        }
        std::cout << "Dectected camera plug out: " << cameraName << ":" << cameraId << std::endl;
    }
}