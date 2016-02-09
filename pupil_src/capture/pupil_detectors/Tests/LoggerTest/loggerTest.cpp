

#include <string>
#include <iostream>
#include "../../../../shared_modules/pycpplog/pycpplogger.h"

int main()
{
    Py_Initialize();
    PyRun_SimpleString("import sys");
    PyRun_SimpleString("sys.path.append(\"../../../../shared_modules/\")");

    auto logger = PyCppLogger( "Awesome Logger" );
    logBasicConfig(); // this just needs to be called if python didn't set up a logger before. see: https://docs.python.org/2/library/logging.html#logging.basicConfig

    logger.error("log Error");

    logger.setLogLevel( PyCppLogger::LogLevel::INFO );
    logger.info("log Error");

    logger.setLogLevel( PyCppLogger::LogLevel::DEBUG );
    logger.debug("log debug ");

    logger.critical("log critical ");
    logger.warn("log warn ");

    auto logger2 = PyCppLogger( "Awesome Logger2" );
    logBasicConfig(); // this just needs to be called if python didn't set up a logger before. see: https://docs.python.org/2/library/logging.html#logging.basicConfig

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
