/*
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2017  Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
*/


#include <string>
#include <iostream>
#include "logger/pycpplogger.h"

int main()
{
    Py_Initialize();

    using namespace pupillabs;
    auto logger = PyCppLogger( "Awesome Logger" );
    logger.basicConfig();// this just needs to be called if python didn't set up a logger before. see: https://docs.python.org/2/library/logging.html#logging.basicConfig
    logger.error("log Error");

    logger.setLogLevel( PyCppLogger::LogLevel::INFO );
    logger.info("log Error");

    logger.setLogLevel( PyCppLogger::LogLevel::DEBUG );
    logger.debug("log debug ");

    logger.critical("log critical ");
    logger.warn("log warn ");

    auto logger2 = PyCppLogger();
    logger2.basicConfig();// this just needs to be called if python didn't set up a logger before. see: https://docs.python.org/2/library/logging.html#logging.basicConfig

    logger2.error("log Error");

    logger2.setLogLevel( PyCppLogger::LogLevel::INFO );
    logger2.info("log Error");

    logger2.setLogLevel( PyCppLogger::LogLevel::DEBUG );
    logger2.debug("log debug ");

    logger2.critical("log critical ");
    logger2.warn("log warn ");

    Py_Finalize();

   return 0;

}
