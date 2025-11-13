/**
 * @file GNSS.cpp
 * Copyright (c) 2023 Dale Giancono All rights reserved.
 *
 * @brief
 * TODO add me
 */
/*****************************************************************************/
/*INLCUDES                                                                   */
/*****************************************************************************/
#include "GNSS.hpp"
#include <thread> // .. for gps_*()
#include <math.h> // for isfinite()
#include <iostream>

/*****************************************************************************/
/*MACROS                                                             */
/*****************************************************************************/
#define MODE_STR_NUM 4

/*****************************************************************************/
/* CLASS DECLARATION                                                                      */
/*****************************************************************************/
GNSS::GNSS()
{
    this->GNSSStopFlag = false;
}

GNSS::~GNSS()
{
    this->GNSSStopFlag = true;
    if (true == thread.joinable())
    {
        thread.join();
    }
}

void GNSS::GNSS_loop(void *arg)
{
    GNSS *self = (GNSS *)arg;
    int ret;

    if (0 != gps_open("localhost", "2947", &self->gpsData))
    {
        std::cout << "GNSS Open error.  Bye, bye" << std::endl;
        return;
    }
    else
    {
        std::cout << "GNSS Opened. " << std::endl;
    }

    (void)gps_stream(&self->gpsData, WATCH_ENABLE | WATCH_JSON, NULL);

    while (false == self->GNSSStopFlag)
    {
        gps_waiting(&self->gpsData, 1000000);
        ret = gps_read(&self->gpsData, NULL, 0);
        bool allgood = true;
        if (-1 == ret)
        {
            std::cout << "GNSS Read Error... " << ret << std::endl;
            continue;
        }

        /* See if the mode is set */
        if (MODE_SET != (MODE_SET & self->gpsData.set))
        {
            // did not even get mode, nothing to see here
            continue;
        }

        /* Update the fix */
        if (0 > self->gpsData.fix.mode || MODE_STR_NUM <= self->gpsData.fix.mode)
        {
            self->gpsData.fix.mode = 0;
        }

        self->mode = self->mode_str[self->gpsData.fix.mode];

        /* Set the time if the time is set */
        if (TIME_SET == (TIME_SET & self->gpsData.set))
        {
            self->timeSeconds = self->gpsData.fix.time.tv_sec;
            self->timeNanoSeconds = self->gpsData.fix.time.tv_nsec;
        }
        else
        {
            std::cout << "Bad Time: " << self->gpsData.fix.time.tv_sec << ", " << self->gpsData.fix.time.tv_nsec << std::endl;
            allgood = false;
        }

        /* Set the lat, lon, if it is valid*/

        if (isfinite(self->gpsData.fix.latitude) && isfinite(self->gpsData.fix.longitude))
        {
            self->latitude = self->gpsData.fix.latitude;
            self->longitude = self->gpsData.fix.longitude;
        }
        else
        {
            std::cout << "Bad Lat, Lon: " << self->gpsData.fix.latitude << ", " << self->gpsData.fix.longitude << std::endl;
            allgood = false;
        }

        if (isfinite(self->gpsData.fix.altMSL))
        {
            self->altitudeMSL = self->gpsData.fix.altMSL;
        }
        else
        {
            std::cout << "Bad Altitude: " << self->gpsData.fix.altMSL << std::endl;
            allgood = false;
        }

        if (true == allgood && nullptr != self->callback)
        {
            self->callback(
                self->latitude,
                self->longitude,
                self->altitudeMSL,
                self->timeSeconds,
                self->timeNanoSeconds,
                self);
        }
    }

    // When you are done...
    (void)gps_stream(&self->gpsData, WATCH_DISABLE, NULL);
    (void)gps_close(&self->gpsData);
    std::cout << "GNSS Closed... " << std::endl;
}

void GNSS::start()
{
    thread = std::thread(GNSS::GNSS_loop, this);
}

void GNSS::start(
    std::function<void(
        double,
        double,
        double,
        double,
        double,
        void *)>
        callback)
{
    this->callback = callback;
    thread = std::thread(GNSS::GNSS_loop, this);
}

void GNSS::get(
    double &latitude,
    double &longitude,
    double &altitudeMSL,
    int64_t &time)
{
    latitude = this->latitude;
    longitude = this->longitude;
    altitudeMSL = this->altitudeMSL;
    time = (int64_t)this->timeSeconds * 1000000000 + (int64_t)this->timeNanoSeconds;
}
