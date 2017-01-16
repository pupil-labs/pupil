#ifndef PYCPPLOGGER_H__
#define PYCPPLOGGER_H__

#include <boost/python.hpp>
#include <string>

namespace pupillabs {

namespace py =  boost::python;

class PyCppLogger{

public:

    PyCppLogger( std::string name ) {
        mLoggingModule = py::import("logging");
        mLogger = mLoggingModule.attr("getLogger")(name);
    };
    PyCppLogger(): PyCppLogger("DefaultLogger"){};

    enum struct LogLevel{ // same levels as in python
        NOTSET = 0,
        DEBUG = 10,
        INFO = 20,
        WARNING = 30,
        ERROR = 40,
        CRITICAL = 50
    };

    void basicConfig(){ mLoggingModule.attr("basicConfig")(); };
    void debug( const std::string& msg ){ mLogger.attr("debug")(msg); };
    void info( const std::string& msg ){  mLogger.attr("info")(msg); };
    void warn( const std::string& msg ){  mLogger.attr("warn")(msg); };
    void error( const std::string& msg ){ mLogger.attr("error")(msg); };
    void critical( const std::string& msg ){ mLogger.attr("critical")(msg); };
    void setLogLevel( LogLevel level ){ mLogger.attr("setLevel")( static_cast<int>(level) ); };

private:

    py::object mLogger;
    py::object mLoggingModule;
};

} // pupillabs

#endif /* end of include guard: PYCPPLOGGER_H__ */
