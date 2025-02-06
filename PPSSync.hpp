#ifndef ALLIEDVISIONALVIUM_PPSSYNC_H_
#define ALLIEDVISIONALVIUM_PPSSYNC_H_

#include "AlliedVisionAlvium.hpp"
#include <mutex>
#include <functional>

class PPSSync
{
public:
    PPSSync();
    ~PPSSync();

    void get(
        int64_t &systemTimestampAtLastPPS,
        int64_t &cameraTimestampAtLastPPS,
        int64_t &systemJitterAtLastPPS,
        int64_t &cameraJitterAtLastCameraPPS);

    enum Line
    {
        LINE0 = 0,
        LINE1 = 1,
        LINE2 = 2,
        LINE3 = 3
    };

    bool enable(
        AlliedVisionAlvium *camera,
        Line line,
        std::function<void(
            int64_t,
            int64_t,
            int64_t,
            int64_t,
            void *)>
            callback,
        void *argument);

private:
    /* Some variables used to GNSS synchonisation */
    int64_t systemTimestampAtLastPPS = 0;
    int64_t cameraTimestampAtLastPPS = 0;
    int64_t previousSystemTimestamp = 0;
    int64_t previousCameraTimestamp = 0;

    int64_t systemJitterAtLastPPS = 0;
    int64_t cameraJitterAtLastCameraPPS = 0;

    std::mutex mutex;

    std::function<void(
        int64_t,
        int64_t,
        int64_t,
        int64_t,
        void *)>
        ppsCallback = nullptr;
    void *ppsCallbackArgument = nullptr;

    static void callback(
        std::string eventName,
        int64_t value,
        time_t systemTimestampSeconds,
        time_t systemTimestampNanoseconds,
        void *arg);
};

#endif // ALLIEDVISIONALVIUM_PPSSYNC_H_
