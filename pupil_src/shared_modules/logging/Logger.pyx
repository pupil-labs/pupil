

from libcpp.string cimport string
import logging

cdef api void logInfo( const string& msg , const string& loggername ):
    logger = logging.getLogger(loggername)
    logger.info(msg)

