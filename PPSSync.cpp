#include "PPSSync.hpp"

PPSSync::PPSSync() {
};

PPSSync::~PPSSync() {
};
void PPSSync::get(
    int64_t &systemTimestampAtLastPPS,
    int64_t &cameraTimestampAtLastPPS,
    int64_t &systemJitterAtLastPPS,
    int64_t &cameraJitterAtLastCameraPPS)
{
    this->mutex.lock();
    systemTimestampAtLastPPS = this->systemTimestampAtLastPPS;
    cameraTimestampAtLastPPS = this->cameraTimestampAtLastPPS;
    systemJitterAtLastPPS = this->systemJitterAtLastPPS;
    cameraJitterAtLastCameraPPS = this->cameraJitterAtLastCameraPPS;
    this->mutex.unlock();
}

bool PPSSync::enable(
    AlliedVisionAlvium *camera,
    Line line,
    std::function<void(
        int64_t,
        int64_t,
        int64_t,
        int64_t,
        void *)>
        callback,
    void *argument)
{
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
    /* Configure the camera for external triggering */

    if (false == camera->setFeature("LineSelector", linestr))
    {
        std::cerr << "Could not set LineSelector" << std::endl;
        return false;
    }
    else if (false == camera->setFeature("LineMode", "Input"))
    {
        std::cerr << "Could not set LineMode" << std::endl;
        return false;
    }
    else if (false == camera->setFeature("LineDebounceMode", "Off"))
    {
        std::cerr << "Could not set LineDebounceMode" << std::endl;
        return false;
    }

    if (false == camera->activateEvent(
                     linestr + "RisingEdge",
                     "Event" + linestr + "RisingEdgeTimestamp",
                     this->callback,
                     this))
    {
        std::cerr << "Could not activate " + linestr + "RisingEdge for PPS event" << std::endl;

        return false;
    }

    this->ppsCallback = callback;
    this->ppsCallbackArgument = argument;

    std::cout
        << "Enabled PPS" << std::endl;

    return true;
}

void PPSSync::callback(
    std::string eventName,
    int64_t value,
    time_t systemTimestampSeconds,
    time_t systemTimestampNanoseconds,
    void *arg)
{
    PPSSync *ppsSync = (PPSSync *)arg;
    int64_t previousCameraTimestamp = ppsSync->cameraTimestampAtLastPPS;
    int64_t previousSystemTimestamp = ppsSync->systemTimestampAtLastPPS;
    int64_t cameraTimestampAtLastPPS = value;
    int64_t systemTimestampAtLastPPS =
        (systemTimestampSeconds * 1000000000) + systemTimestampNanoseconds;
    int64_t cameraJitterAtLastCameraPPS =
        (cameraTimestampAtLastPPS - previousCameraTimestamp - 1000000000) / 1000;
    int64_t systemJitterAtLastPPS =
        (systemTimestampAtLastPPS - previousSystemTimestamp - 1000000000) / 1000;

    ppsSync->mutex.lock();
    ppsSync->previousCameraTimestamp = previousCameraTimestamp;
    ppsSync->previousSystemTimestamp = previousSystemTimestamp;
    ppsSync->cameraTimestampAtLastPPS = cameraTimestampAtLastPPS;
    ppsSync->systemTimestampAtLastPPS = systemTimestampAtLastPPS;
    ppsSync->cameraJitterAtLastCameraPPS = cameraJitterAtLastCameraPPS;
    ppsSync->systemJitterAtLastPPS = systemJitterAtLastPPS;
    ppsSync->mutex.unlock();

    if (ppsSync->ppsCallback != nullptr)
    {
        ppsSync->ppsCallback(
            cameraTimestampAtLastPPS,
            systemTimestampAtLastPPS,
            cameraJitterAtLastCameraPPS,
            systemJitterAtLastPPS,
            ppsSync->ppsCallbackArgument);
    }

    return;
}
