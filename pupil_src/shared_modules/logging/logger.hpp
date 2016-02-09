


#ifndef LOGGER_H__
#define LOGGER_H__



#include <string>
#include "logger.h"



void logInfo( const std::string& msg, const std::string& logger = "cpplogger" ){
    logInfo( msg.c_str() , logger.c_str() );
}


#endif /* end of include guard: LOGGER_H__ */
