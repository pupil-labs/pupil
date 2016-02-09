/*
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2016  Pupil Labs

 Distributed under the terms of the GNU Lesser General Public License (LGPL v3.0).
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
*/

/*
    This is a small wrapper for the pycpplog.pyx file.
    This makes it easier to use in C++ source code, because it keeps track of the logger state
*/


#ifndef PYCPPLOGGER_H__
#define PYCPPLOGGER_H__


#include <string>
#include "pycpplog_api.h"

class PyCppLogger{

public:

    PyCppLogger( std::string name ) : mLoggerName(name) {
        import_pycpplog__pycpplog();
    };

    enum struct LogLevel{ // same levels as in python
        NOTSET = 0,
        DEBUG = 10,
        INFO = 20,
        WARNING = 30,
        ERROR = 40,
        CRITICAL = 50
    };

    void debug( const std::string& msg ){ logDebug(msg, mLoggerName); };
    void info( const std::string& msg ){ logInfo(msg, mLoggerName); };
    void warn( const std::string& msg ){ logWarn(msg, mLoggerName); };
    void error( const std::string& msg ){ logError(msg, mLoggerName); };
    void critical( const std::string& msg ){ logCritical(msg, mLoggerName); };
    void setLogLevel( LogLevel level ){ logLevel(static_cast<int>(level), mLoggerName); };

private:
    PyCppLogger(){};

    std::string mLoggerName;



};


#endif /* end of include guard: PYCPPLOGGER_H__ */
