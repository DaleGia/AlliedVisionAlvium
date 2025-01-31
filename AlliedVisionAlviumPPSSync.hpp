#include <atomic>
#include "VmbCPP/VmbCPP.h"
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

class AlliedVisionPPSSync
{
public:
    AlliedVisionPPSSync();
    ~AlliedVisionPPSSync();

    enum Line
    {
        Line0 = 0,
        Line1 = 1,
        Line2 = 2,
        Line3 = 3
    };

    bool enable(AlliedVisionPPSSync::Line line);

private:
    std::atomic<int64_t> lastSystemTimeAtCameraPPS = 0;
    std::atomic<int64_t> lastSystemTimeJitter = 0;
    std::atomic<int64_t> lastCameraPPSTimestamp = 0;
    std::atomic<int64_t> lastCameraTimeJitter = 0;
};
