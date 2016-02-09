

#include <string>
#include <iostream>
#include "../../../../shared_modules/logging/logger_api.h"

int main()
{
    Py_Initialize();
    import_logging__logger();
    std::cout << "test2"  << std::endl;
    std::string msg = "log this!";
    std::string logger = "cpplogger";
    logInfo(msg, logger ) ;
    std::cout << "test3"  << std::endl;

    Py_Finalize();

   return 0;

}
