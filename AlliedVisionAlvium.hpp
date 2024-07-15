#ifndef ALLIEDVISIONALVIUM_H_
#define ALLIEDVISIONALVIUM_H_

#include <iostream>
#include <opencv2/opencv.hpp>
#include "VmbCPP/VmbCPP.h"
#include "VmbImageTransform/VmbTransform.h"

/**
 * \brief IFrameObserver implementation for asynchronous image acquisition
 */
class FrameObserver : public VmbCPP::IFrameObserver
{
    public:
        FrameObserver(VmbCPP::CameraPtr camera) : IFrameObserver(camera)
        {
        };

        FrameObserver(
            VmbCPP::CameraPtr camera,
            std::function<void(cv::Mat, uint64_t, uint64_t, void*)> imageCallback,
            void* arg) : IFrameObserver(camera), callback(imageCallback), argument(arg)
        {
            
        };

        void FrameReceived(const VmbCPP::FramePtr frame)
        {
            VmbError_t err;
            int openCvType;
            VmbPixelFormatType format;
            VmbFrameStatusType status = VmbFrameStatusComplete;
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

            err = frame->GetReceiveStatus(status);

            if(VmbErrorSuccess != err)
            {
                std::cerr << "Could not get frame status" << std::endl;
            }
            else if(VmbFrameStatusComplete != status)
            {
                switch(status)
                {
                    case VmbFrameStatusIncomplete: 
                    {
                        std::cerr << "Frame incomplete. Try a slower frame rate" << std::endl;
                        return;
                    }
                    case VmbFrameStatusTooSmall: 
                    {
                        std::cerr << "Frame too small..." << std::endl;
                        return;
                    }
                    case VmbFrameStatusInvalid: 
                    {
                        std::cerr << "Frame invalid..." << std::endl;
                        return;
                    }
                }
            }

            frame->GetPixelFormat(format);
            frame->GetHeight(height);
            frame->GetWidth(width);
            frame->GetTimestamp(timestamp);
            frame->GetFrameID(frameID);
            frame->GetBufferSize(bufferSize);
            frame->GetBuffer(data);
            switch(format)
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
                    destinationImage.Data = malloc(width*height*2);
                    if(nullptr == destinationImage.Data)
                    {   
                        std::cerr << "Could not create destination buffer for unpacking" << std::endl;
                        return;
                    }


                    VmbError_t error = VmbSetImageInfoFromPixelFormat(
                        VmbPixelFormatMono12p,
                        width,
                        height,
                        &sourceImage);
                    if(VmbErrorSuccess != error)
                    {
                        std::cerr << "Could not create source image info for unpacking: " << error << std::endl;
                        return;
                    }
                    error = VmbSetImageInfoFromInputParameters(
                        VmbPixelFormatMono12,
                        width,
                        height,
                        VmbPixelLayoutMono,
                        16,
                        &destinationImage);
                    if(error != VmbErrorSuccess)
                    {
                        std::cerr << "Could not create destination image info for unpacking: " << error << std::endl;
                    }
                    error = VmbImageTransform(
                        &sourceImage, 
                        &destinationImage, 
                        NULL, 
                        0);
                    if(error != VmbErrorSuccess)
                    {
                        std::cerr << "Could not unpack image: " << error << std::endl;
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
                    return;
                }
            }

            /* returns the frame buffer back to the queue */
            m_pCamera->QueueFrame(frame);
            if(nullptr != this->callback)
            {
                this->callback(image, timestamp, frameID, this->argument);
            }
        };

    private:
        std::function<void(cv::Mat, uint64_t, uint64_t, void*)> callback = nullptr;
        void* argument = nullptr;
};

class AlliedVisionAlvium
{
    public:
        AlliedVisionAlvium();
        ~AlliedVisionAlvium();
        
        bool connect(std::string cameraName);
        bool disconnect(void);
        bool isCameraOpen(void);

        bool getFeature(
            std::string featureName, 
            std::string &featureValue);
        bool setFeature(
            std::string featureName, 
            std::string featureValue);
        bool runCommand(std::string command);
        


        bool startAcquisition(
            int bufferCount, 
            std::function<void(cv::Mat, uint64_t, uint64_t, void*)> newFrameCallback, 
            void* arg);
        bool stopAcquisition(void);
        bool getSingleFrame(cv::Mat &buffer, uint32_t timeoutMs);

        std::string getName(void);
        bool getId(std::string &buffer);

        bool getFrameRateHz(std::string &buffer);
        bool getExposureUs(std::string &buffer);
        bool getGainDb(std::string &buffer);
        bool getBitDepth(std::string &buffer);
        bool getTemperature(std::string &buffer);

        bool setFrameRateHz(std::string buffer);
        bool setExposureUs(std::string buffer);
        bool setGainDb(std::string buffer);
        bool setBitDepth(std::string buffer);
        bool setDeviceThroughputLimit(std::string buffer);

        bool loadConfiguration1(void);
        bool saveConfiguration1(void);
        bool loadSettingsFile(std::string filepath);
        bool saveSettingsFile(std::string filepath);

    private:
        VmbCPP::CameraPtr camera;
        std::string name;
        bool cameraOpen = false;

};

AlliedVisionAlvium::AlliedVisionAlvium()
{
}

AlliedVisionAlvium::~AlliedVisionAlvium()
{
    try
    {
        if(cameraOpen)
        {
            this->disconnect();
        }
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
    }
    

}

bool AlliedVisionAlvium::connect(std::string cameraName)
{
    VmbCPP::VmbSystem &vimbax  = 
        VmbCPP::VmbSystem::GetInstance(); 

    VmbErrorType err;
    this->name = cameraName;
    /* Start the API, get and open cameras */
    err = vimbax.Startup();
    if(err == VmbErrorAlready)
    {
        /* Do not do nothin'*/
    }
    else if(err != VmbErrorSuccess)
    {
        std::cerr << "Unable to start up vimbax " << 
        cameraName << ": " << err << std::endl;
        this->cameraOpen = false;
        return false;
    }

    err = vimbax.OpenCameraByID(
        cameraName.c_str(),
        VmbAccessModeExclusive,
        this->camera);
    if(err != VmbErrorSuccess)
    {
        std::cerr << "Unable to connect to camera " << 
        cameraName << ": " << err << std::endl;
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
    if(err != VmbErrorSuccess)
    {
        std::cerr << "Could not disconnect from camera..." << std::endl;
    }

    this->cameraOpen = false;
    return this->cameraOpen;
}

bool AlliedVisionAlvium::getSingleFrame(cv::Mat &buffer, uint32_t timeoutMs)
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
    if(VmbErrorSuccess != err)
    {
        std::cerr << "Could not get single frame: " << err << std::endl;
        return false;
    }

    err = frame->GetReceiveStatus(status);

    if(VmbErrorSuccess != err)
    {
        std::cerr << "Could not get frame status" << std::endl;
        return false;
    }
    else if(VmbFrameStatusComplete != status)
    {
        switch(status)
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
    frame->GetFrameID(frameID);
    frame->GetBufferSize(bufferSize);
    frame->GetBuffer(data);
    switch(format)
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
            destinationImage.Data = malloc(width*height*2);
            if(nullptr == destinationImage.Data)
            {   
                std::cerr << "Could not create destination buffer for unpacking" << std::endl;
                return false;
            }


            VmbError_t error = VmbSetImageInfoFromPixelFormat(
                VmbPixelFormatMono12p,
                width,
                height,
                &sourceImage);
            if(VmbErrorSuccess != error)
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
            if(error != VmbErrorSuccess)
            {
                std::cerr << "Could not create destination image info for unpacking: " << error << std::endl;
                return false;
            }
            error = VmbImageTransform(
                &sourceImage, 
                &destinationImage, 
                NULL, 
                0);
            if(error != VmbErrorSuccess)
            {
                std::cerr << "Could not unpack image: " << error << std::endl;
                return false;
            }
            image = cv::Mat(
                height,
                width, 
                openCvType, 
                destinationImage.Data);
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
    free(destinationImage.Data);

    return true;
}
bool AlliedVisionAlvium::startAcquisition(
    int bufferCount, 
    std::function<void(cv::Mat, uint64_t, uint64_t, void*)> newFrameCallback, 
    void* arg)
{
    VmbErrorType err;
    err = camera->StartContinuousImageAcquisition(
         bufferCount, 
         VmbCPP::IFrameObserverPtr(new FrameObserver(camera, newFrameCallback, arg)));
    if(VmbErrorSuccess != err)
    {
        return false;
    }
    
    return true;
}

bool AlliedVisionAlvium::stopAcquisition(void)
{
    VmbErrorType err;
    err = camera->StopContinuousImageAcquisition();
    if(VmbErrorSuccess != err)
    {
        return false;
    }
    
    return true;
}

bool AlliedVisionAlvium::isCameraOpen(void)
{
    return this->cameraOpen;
}

std::string AlliedVisionAlvium::getName(void)
{    
    VmbErrorType err;
    return this->name;
}

bool AlliedVisionAlvium::getId(std::string &buffer)
{
    VmbErrorType err;

    if(false == this->isCameraOpen())
    {
        return false;
    }
    err = this->camera->GetID(buffer);
    
    if(VmbErrorSuccess != err)
    {
        return false;
    }

    return true;
}

bool AlliedVisionAlvium::getFrameRateHz(std::string &buffer)
{
    if(false == this->getFeature("AcquisitionFrameRate", buffer))
    {
        return false;
    }

    return true;
}

bool AlliedVisionAlvium::getExposureUs(std::string &buffer)
{
    if(false == this->getFeature("ExposureTime", buffer))
    {
        return false;
    }
    else
    {
        return true;
    }
}

bool AlliedVisionAlvium::getGainDb(std::string &buffer)
{
    if(false == this->getFeature("Gain", buffer))
    {
        return false;
    }
    else
    {
        return true;
    }
}

bool AlliedVisionAlvium::setFrameRateHz(std::string buffer)
{
    if(false == this->setFeature("AcquisitionFrameRateEnable", "true"))
    {   
    }
    
    if(false == this->setFeature("AcquisitionFrameRate", buffer))
    {
        return false;
    }
    
    return true;
}

bool AlliedVisionAlvium::setExposureUs(std::string buffer)
{
    if(false == this->setFeature("ExposureTime", buffer))
    {
        return false;
    }
    else
    {
        return true;
    }
}

bool AlliedVisionAlvium::setGainDb(std::string buffer)
{
    if(false == this->setFeature("Gain", buffer))
    {
        return false;
    }
    else
    {
        return true;
    }
}


bool AlliedVisionAlvium::getBitDepth(std::string &buffer)
{
    if(false == this->getFeature("PixelFormat", buffer))
    {
        return false;
    }
    else
    {
        return true;
    }
}

bool AlliedVisionAlvium::getTemperature(std::string &buffer)
{
    if(false == this->getFeature("DeviceTemperature", buffer))
    {
        return false;
    }
    else
    {
        return true;
    }
}


bool AlliedVisionAlvium::setBitDepth(std::string buffer)
{
    if(false == this->setFeature("PixelFormat", buffer))
    {
        return false;
    }
    else
    {
        return true;
    }
}

bool AlliedVisionAlvium::setDeviceThroughputLimit(std::string buffer)
{
    if(false == this->setFeature("DeviceLinkThroughputLimitMode", "Off"))
    {
        std::cerr << "AlliedVisionAlvium: Could not set DeviceLinkThroughputLimitMode Off" << std::endl;
        return false;
    }
    if(false == this->setFeature("DeviceLinkThroughputLimit", buffer))
    {
        std::cerr << "AlliedVisionAlvium: Could not set DeviceLinkThroughputLimit" << std::endl;
        return false;
    }
    
    if(false == this->setFeature("DeviceLinkThroughputLimitMode", "On"))
    {
        std::cerr << "AlliedVisionAlvium: Could not set DeviceLinkThroughputLimitMode On" << std::endl;
        return false;
    }

    return true;
}


bool AlliedVisionAlvium::loadSettingsFile(std::string filepath)
{
    VmbError_t err;
    VmbFeaturePersistSettings_t settings;
    settings.loggingLevel = VmbLogLevelAll;
    settings.maxIterations = 10;
    settings.persistType = VmbFeaturePersistAll;
    settings.modulePersistFlags = VmbModulePersistFlagsAll;
    err = this->camera->LoadSettings(filepath.c_str(), &settings);

    if(VmbErrorSuccess != err)
    {
        std::cout << "Error loading settings: " << err << " - " << filepath << std::endl;
        return false;
    }

    return true;
}

bool AlliedVisionAlvium::saveSettingsFile(std::string filepath)
{
    VmbError_t err;
    VmbFeaturePersistSettings_t settings;
    settings.loggingLevel = VmbLogLevelError;
    settings.maxIterations = 10;
    settings.persistType = VmbFeaturePersistAll;
    settings.modulePersistFlags = VmbModulePersistFlagsAll;
    err = this->camera->SaveSettings(filepath.c_str(), &settings);

    if(VmbErrorSuccess != err)
    {
        std::cout << "Error saving settings: " << err << " - " << filepath << std::endl;
        return false;
    }

    return true;
}

bool AlliedVisionAlvium::loadConfiguration1(void)
{
    if(false == this->setFeature("UserSetDefault", "UserSet1"))
    {
        return false;
    }
    
    if(false == this->setFeature("UserSetSelector", "UserSet1"))
    {
        return false;
    }

    if(false == this->runCommand("UserSetLoad"))
    {
        return false;
    }

    return true;
}

bool AlliedVisionAlvium::saveConfiguration1(void)
{
    if(false == this->setFeature("UserSetDefault", "UserSet1"))
    {
        return false;
    }
    
    if(false == this->setFeature("UserSetSelector", "UserSet1"))
    {
        return false;
    }

    if(false == this->runCommand("UserSetSave"))
    {
        return false;
    }

    return true;
}

bool AlliedVisionAlvium::setFeature(
    std::string featureName, 
    std::string featureValue)
{
    VmbCPP::FeaturePtr feature;
    VmbError_t error;
    VmbFeatureDataType dataType;
      
    error = this->camera->GetFeatureByName(featureName.c_str(), feature);
    if(VmbErrorSuccess != error)
    {         
        std::cerr << "Could not get feature " << featureName << std::endl;
        return false;
    }
   
    error = feature->GetDataType(dataType);
    if(VmbErrorSuccess != error)
    {      
        std::cerr << "Could not get feature " << featureName << " datatype" << std::endl;   
        return false;
    }   

    if(VmbFeatureDataInt == dataType)
    {
        error = feature->SetValue(std::stoi(featureValue));

    }
    else if(VmbFeatureDataFloat == dataType)
    {
        error = feature->SetValue(std::stof(featureValue));
    }
    else if(VmbFeatureDataEnum == dataType)
    {
        error = feature->SetValue(featureValue.c_str());
    }
    else if(VmbFeatureDataString == dataType)
    {
        error = feature->SetValue(featureValue.c_str());
    }
    else if(VmbFeatureDataBool == dataType)
    {
        if(featureValue == "false")
        {
            error = feature->SetValue(false);
        }
        else if(featureValue == "true")
        {
            error = feature->SetValue(true);
        }
    }
    else if(VmbFeatureDataCommand == dataType)
    {
        feature->RunCommand();
    }
    else
    {
        std::cerr << "Unknown feature datatype: " << dataType << " for " << 
        featureName << std::endl;   
        return false;
    }

    if(VmbErrorSuccess != error)
    {      
        std::cerr << "Could not set feature " << featureName << ": " << error << std::endl;   
        return false;
    }

   return true;
}

bool AlliedVisionAlvium::getFeature(
    std::string featureName, 
    std::string &featureValue)
{
    VmbCPP::FeaturePtr feature;
    VmbError_t error;
    VmbFeatureDataType dataType;
      
    error = this->camera->GetFeatureByName(featureName.c_str(), feature);
    if(VmbErrorSuccess != error)
    {         
        std::cerr << "Could not get feature " << featureName << ": " << error << std::endl;
        return false;
    }
    error = feature->GetDataType(dataType);
    if(VmbErrorSuccess != error)
    {      
        std::cerr << "Could not get feature " << featureName << " datatype" << std::endl;   
        return false;
    }   

    if(VmbFeatureDataInt == dataType)
    {
        VmbInt64_t data;
        error = feature->GetValue(data);
        if(VmbErrorSuccess != error)
        {      
            std::cerr << "Could not get feature value" << featureName << ": " << error << std::endl;
            return false;
        }
        featureValue =  std::to_string(data); 

    }
    else if(VmbFeatureDataFloat == dataType)
    {
        double data;
        error = feature->GetValue(data);
        if(VmbErrorSuccess != error)
        {      
            std::cerr << "Could not get feature value" << featureName << ": " << error << std::endl;
            return false;
        }
        featureValue =  std::to_string(data); 
    }
    else if(VmbFeatureDataString == dataType)
    {
        std::string data;
        error = feature->GetValue(data);
        if(VmbErrorSuccess != error)
        {      
            std::cerr << "Could not get feature value" << featureName << ": " << error << std::endl;
            return false;
        }
        featureValue = data; 
    }
    else if(VmbFeatureDataBool == dataType)
    {
        bool data;
        error = feature->GetValue(data);
        if(VmbErrorSuccess != error)
        {      
            std::cerr << "Could not get feature value" << featureName << ": " << error << std::endl;
            return false;
        }
        featureValue =  std::to_string(data); 
    }
    else if(VmbFeatureDataEnum == dataType)
    {
        std::string data;
        error = feature->GetValue(data);
        if(VmbErrorSuccess != error)
        {      
            std::cerr << "Could not get feature value" << featureName << ": " << error << std::endl;
            return false;
        }
        featureValue = data; 
    }
    else
    {
        std::cerr << "Unknown feature datatype: " << dataType << " for " << 
        featureName << std::endl;   
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
    if(VmbErrorSuccess != error)
    {         
        std::cerr << "Could not get feature " << command << ": " << error << std::endl;
        return false;
    }

    error = feature->RunCommand();
    if(VmbErrorSuccess != error)
    {         
        std::cerr << "Could not run command " << command << ": " << error << std::endl;
        return false;
    }

    bool ret;
    while(true)
    {
        error = feature->IsCommandDone(ret);

        if(VmbErrorSuccess != error)
        {         
            std::cerr << "Could not wait for command " + command << ": " << error << std::endl;
            return false;
        }
        else if(true == ret)
        {
            break;
        }
    }

    return true;
}

#endif // ALLIEDVISIONALVIUM_H_
