

from libcpp.string cimport string
import logging

cdef api void logBasicConfig():
    logging.basicConfig()

cdef api void logInfo( const string& msg , const string& loggername ):
    logger = logging.getLogger(loggername)
    logger.info(msg)

cdef api void logWarn( const string& msg , const string& loggername ):
    logger = logging.getLogger(loggername)
    logger.warn(msg)

cdef api void logError( const string& msg , const string& loggername ):
    logger = logging.getLogger(loggername)
    logger.error(msg)

cdef api void logLevel( int level , const string& loggername ):
    logger = logging.getLogger(loggername)
    logger.setLevel(level)
