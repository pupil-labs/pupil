'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2016  Pupil Labs

 Distributed under the terms of the GNU Lesser General Public License (LGPL v3.0).
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''


from libcpp.string cimport string
import logging

cdef api void logBasicConfig():
    logging.basicConfig()

cdef api void logDebug( const string& msg , const string& loggername ):
    logger = logging.getLogger(loggername)
    logger.debug(msg)

cdef api void logInfo( const string& msg , const string& loggername ):
    logger = logging.getLogger(loggername)
    logger.info(msg)

cdef api void logWarn( const string& msg , const string& loggername ):
    logger = logging.getLogger(loggername)
    logger.warn(msg)

cdef api void logError( const string& msg , const string& loggername ):
    logger = logging.getLogger(loggername)
    logger.error(msg)

cdef api void logCritical( const string& msg , const string& loggername ):
    logger = logging.getLogger(loggername)
    logger.error(msg)

cdef api void logLevel( int level , const string& loggername ):
    logger = logging.getLogger(loggername)
    logger.setLevel(level)
