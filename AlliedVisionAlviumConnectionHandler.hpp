#include <iostream>
#include <VmbCPP/VmbCPP.h>

class AlliedVisionAlviumConnectionHandler : public VmbCPP::ICameraListObserver
{
    public:
        AlliedVisionAlviumConnectionHandler(
            std::function<void(std::string&, std::string&)> connectedCallback,
            std::function<void(std::string&, std::string&)> disconnectedCallback) : 
            VmbCPP::ICameraListObserver(),
            cameraConnectedCallback(connectedCallback),
            cameraDisconnectedCallback(disconnectedCallback)
        {

        };

        ~AlliedVisionAlviumConnectionHandler()
        {
            VmbCPP::VmbSystem &vimbax  = 
                VmbCPP::VmbSystem::GetInstance();
            vimbax.Shutdown();      
        };

        void start(void)
        {
            VmbErrorType err;
            VmbCPP::CameraPtrVector cameras;   
            std::string cameraName;
            std::string cameraId;

            try
            {
                VmbCPP::VmbSystem &vimbax  = 
                    VmbCPP::VmbSystem::GetInstance();
                err = vimbax.Startup();
                
                /* Generate the initial camera list and call the connection 
                    callback just because it is convinient
                */
                if(VmbErrorSuccess != err)
                {
                    std::cerr << "Could not start VimbaX... " << err << std::endl;
                }
                else if(VmbErrorSuccess != vimbax.GetCameras(cameras))
                {
                    std::cerr << "Could not get camera list..." << err << std::endl;
                }
                else if(0 == cameras.size())
                {
                    std::cout << "No Cameras connected..." << std::endl;
                    vimbax.RegisterCameraListObserver(VmbCPP::ICameraListObserverPtr(this)); 
                }
                else if(VmbErrorSuccess != cameras[0]->GetName(cameraName))
                {
                    std::cerr << "Could not get camera name..." << err << std::endl;
                }
                else if(VmbErrorSuccess != cameras[0]->GetID(cameraId))
                {
                    std::cerr << "Could not get camera id..." << err << std::endl;
                }
                else
                {
                    std::cout << "Camera already connected..." << std::endl;
                    this->cameraConnectedCallback(cameraName, cameraId);
                    vimbax.RegisterCameraListObserver(VmbCPP::ICameraListObserverPtr(this)); 
                }
            }
            catch(const std::exception& e)
            {
                std::cerr << e.what() << std::endl;
            }
            
        }

        void CameraListChanged(
        VmbCPP::CameraPtr pCam, 
        VmbCPP::UpdateTriggerType reason)
        {
            if(reason == VmbCPP::UpdateTriggerPluggedIn)
            {
                VmbErrorType err;
                VmbCPP::CameraPtrVector cameras;   
                std::string cameraName;
                std::string cameraId;

                err = pCam->GetName(cameraName);
                if(err != VmbErrorSuccess)
                {
                    std::cerr << "Could not get camera name..." << std::endl;
                }

                err = pCam->GetID(cameraId);
                if(err != VmbErrorSuccess)
                {
                    std::cerr << "Could not get camera id..." << std::endl;
                }
                std::cout << "Dectected camera plug in: " << cameraName << ":" << cameraId << std::endl;

                this->cameraConnectedCallback(cameraName, cameraId);
            }
            else if(reason == VmbCPP::UpdateTriggerPluggedOut)
            {
                VmbErrorType err;
                VmbCPP::CameraPtrVector cameras;   
                std::string cameraName;
                std::string cameraId;
                err = pCam->GetName(cameraName);
                if(err != VmbErrorSuccess)
                {
                    std::cerr << "Could not get camera name..." << std::endl;
                }

                err = pCam->GetID(cameraId);
                if(err != VmbErrorSuccess)
                {
                    std::cerr << "Could not get camera id..." << std::endl;
                }
                std::cout << "Dectected camera plug out: " << cameraName << ":" << cameraId << std::endl;

                this->cameraDisconnectedCallback(cameraName, cameraId);
            }
        }

    private:
        std::function<void(std::string&, std::string&)> cameraConnectedCallback = nullptr;
        std::function<void(std::string&, std::string&)> cameraDisconnectedCallback = nullptr;
};