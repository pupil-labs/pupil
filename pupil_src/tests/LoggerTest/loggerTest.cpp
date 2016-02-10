

#include <string>
#include <iostream>
#include "../../shared_cpp/Logger/PyCpplogger.h"

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
