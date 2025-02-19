/**
 * @file GNSS.hpp
 * Copyright (c) 2023 Dale Giancono All rights reserved.
 *
 * @brief
 * TODO add me
 */
/*****************************************************************************/
/*INLCUDES                                                                   */
/*****************************************************************************/
#include <gps.h>  // .. for gps_*()
#include <thread> // .. for gps_*()
#include <math.h> // for isfinite()
#include <functional>
#include <atomic>
#include <string>

/*****************************************************************************/
/*MACROS                                                             */
/*****************************************************************************/
#define MODE_STR_NUM 4

/*****************************************************************************/
/* CLASS DECLARATION                                                                      */
/*****************************************************************************/
class GNSS
{
public:
    GNSS();
    ~GNSS();

    void start(void);
    void start(
        std::function<void(
            double,
            double,
            double,
            double,
            double,
            void *)>
            callback);
    void get(
        double &latitude,
        double &longitude,
        double &altitudeMSL,
        int64_t &time);

private:
    static void GNSS_loop(void *arg);

    const char *mode_str[MODE_STR_NUM] =
        {
            "n/a",
            "None",
            "2D",
            "3D"};
    std::atomic<bool> GNSSStopFlag;
    std::thread thread;
    struct gps_data_t gpsData;

    double latitude = 0.0;
    double longitude = 0.0;
    double altitudeMSL = 0.0;
    double timeSeconds = 0.0;
    double timeNanoSeconds = 0.0;
    double timeUncertantitySeconds = 0.0;

    std::string mode;

    std::function<void(
        double,
        double,
        double,
        double,
        double,
        void *)>
        callback = nullptr;
};
