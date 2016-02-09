

#include <string>
#include <iostream>
#include "../../../../shared_modules/pycpplog/pycpplog_api.h"

int main()
{
    Py_Initialize();
    PyRun_SimpleString("import sys");
    PyRun_SimpleString("sys.path.append(\"../../../../shared_modules/\")");
    import_pycpplog__pycpplog();
    loggingBasicConfig();
    std::string msg = "log this!";
    std::string msgInfo = "log info!";
    std::string logger = "testLogger";
    logInfo(msgInfo, logger ) ;
    logWarn(msg, logger ) ;
    logError(msg, logger );
    logLevel(9 , logger);
    logInfo(msgInfo, logger ) ;
    Py_Finalize();

   return 0;

}
