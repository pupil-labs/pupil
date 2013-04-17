/* by sluggo, sluggo@dm9.se ( http://dm9.se )
 * to use libuvcc from source you need to import the IOKit framework
 */

#ifndef __LIBUVCC_H__
#define __LIBUVCC_H__

#include <CoreFoundation/CoreFoundation.h>
#include <IOKit/usb/IOUSBLib.h>
#include "libuvcc-defs.h"

/* name stored in uvccCam->regName if no name is found, edit as you please */
#define UVCC_UNKNOWN_CAM_REG_NAME	"Unknown camera"

/** \defgroup main Main
 * Functions to get, release, open, close and send requests to cams.
 * @{ */
/** \defgroup qt QT-kit helpers
 * Helper functions to easily incorporate uvcc into QT-kit projects.
 */

 /**
 * Init uvcc.
 * \return 0 on success
 */
int uvccInit();
/**
 * Exits uvcc.
 */
void uvccExit();
/**
 * uvcc version (CFStringRef can be safely casted to NSString*).
 * \return uvcc version string
 */
CFStringRef uvccVersion();
/**
 * Populates list with all usb connected cams that have a video control interface. List needs to be released by \ref uvccReleaseCamList(), note that list of length 0 is not allocated and should therefor not be released.
 * \param list cam double pointer to hold cams
 * \return Length of list or -1
 */
int uvccGetCamList(uvccCam ***list);
/** \ingroup qt
 * Get cam with QT-kit unique id. Cam needs to be released by \ref uvccReleaseCam().
 * \param uId QT-kit id
 * \param cam cam pointer to set
 * \return 0 on success != 0 otherwise
 */
int uvccGetCamWithQTUniqueID(CFStringRef uId, uvccCam **cam);
/**
 * Populates list with all usb connected cams that have are identified by modelID. List needs to be released by \ref uvccReleaseCamList(), note that list of length 0 is not allocated and should therefor not be released.
 * \param mId modelID identifying cam
 * \param list cam double pointer to hold cams
 * \return Length of list or -1
 */
int uvccGetCamsWithModelID(struct uvccModelID *mID, uvccCam ***list);
/**
 * Release and free list of cams.
 * \param list list of cams
 * \param len length of list
 */
void uvccReleaseCamList(uvccCam **list, int len);
/**
 * Release single cam.
 * \param cam
 */
void uvccReleaseCam(uvccCam *cam);
/** \ingroup qt
 * The QT-kit unique id of the cam, as [QTCaptureDevice uniqueID]. Note that returned CFStringRef (NSString*) must be released.
 * \param cam
 * \return CFStringRef to id or NULL on error
 */
CFStringRef uvccCamQTUniqueID(uvccCam *cam);
/** \defgroup cam_str Cam strings
 * Functions to retrieve strings from a cam.
 * @{ */
/**
 * Gets the manufacturer string (can be empty). Note that returned CFStringRef (NSString*) must be released.
 * \param cam
 * \return CFStringRef to string or NULL on error
 */
CFStringRef uvccCamManufacturer(uvccCam *cam);
/**
 * Params and return as \ref uvccCamManufacturer().
 */
CFStringRef uvccCamProduct(uvccCam *cam);
/**
 * Params and return as \ref uvccCamManufacturer().
 */
CFStringRef uvccCamSerialNumber(uvccCam *cam);
/** @} */
/**
 * Convert from UniChar to char.
 * \param src source
 * \param dst destination
 * \param len number of UniChars to convert
 * \param errChar char to set if src[i] is non-ascii or 0 to skip them
 * \return Length of dst
 */
int uvccUni2Char(UniChar *src, char *dst, int len, int errChar);
/**
 * Open cam to be able send requests to it, this takes control over it and does not let any other process send requests while you have it opened.
 * \param cam
 * \return 0 on success != 0 otherwise
 */
int uvccOpenCam(uvccCam *cam);
/**
 * Close cam.
 * \param cam
 */
void uvccCloseCam(uvccCam *cam);
/**
 * Send predefined uvcc request to cam.
 * \param cam
 * \param bRequest uvc request to send, valid values are those in \ref uvc_bRequest
 * \param uvccReq uvcc request to send, valid values are those in \ref uvccRequest
 * \param pData data to send or buf to receive data in
 * \return 0 on success or error code
 */
int uvccSendRequest(uvccCam *cam, uint8_t bRequest, enum uvccRequest uvccReq, void *pData);
/**
 * For the brave.. send a raw request on the default control pipe of the video ctrl interface of the cam!
 * \param cam
 * \param request to send
 * \return 0 on success or error code
 */
int uvccSendRawRequest(uvccCam *cam, IOUSBDevRequest *request);
/**
 * Kind of wrapper for UVC_GET_INFO. Result can be &'ed with \ref uvcc_info_flags to see what operations are supported.
 * \param cam
 * \param uvccReq to get info about
 * \param pData buf to hold result 
 * \return 0 on success or error code
 */
int uvccSendInfoRequest(uvccCam *cam, enum uvccRequest uvccReq, int8_t *pData);
/**
 * Retrieves description of last error together with the function that caused it and the corresponding OS X error code. Note that returned CFStringRef (NSString*) must be released.
 * \return CFStringRef to description or NULL on error
 */
CFStringRef uvccErrorStr();
/** \defgroup hlp Non-standard helpers
 * Helper functions to handle cams that don't quite follow the UVC standard.
 * @{ */
/**
 * Produces exposure time values in increasing order that can be used with Microsoft LifeCams. The array have to be freed by the caller.
 * \param min min value from cam
 * \param max max value from cam
 * \param list buffer to hold the array
 * \return length of array or < 0 on error
 */
int uvccMSLifeCamExposureTimeSpan(uint32_t min, uint32_t max, uint32_t **list);
/**
 * Convert exposure time value reported by (UVC_GET_CUR) Microsoft LifeCam to corresponding index in array created by \ref uvccMSLifeCamExposureTimeSpan().
 * \param value value from cam
 * \param msList array from \ref uvccMSLifeCamExposureTimeSpan()
 * \param listLength length of list
 * \return index in list that holds the value or closest match
 */
int uvccExposureTimeToMsLifeCamValue(uint32_t value, uint32_t *msList, int listLength);
/** @} */

/** @} */

/**
 * \defgroup reqs Request Wrappers
 * Wrapper functions to easily send requests to a cam.
 * @{ */
/** Params and return as \ref uvccRWAutoFocus(), UVC_GET_CUR result can be found in \ref uvcc_scanning_mode_returns. */
int uvccRWScanningMode(uvccCam *cam, uint8_t request, uint8_t *value);
/** Params and return as \ref uvccRWAutoFocus(), UVC_GET_CUR result can be found in \ref uvcc_et_mode_returns. */
int uvccRWAutoExposureMode(uvccCam *cam, uint8_t request, int8_t *value);
/** Params and return as \ref uvccRWAutoFocus(), UVC_GET_CUR result can be found in \ref uvcc_et_prio_returns.  */
int uvccRWAutoExposurePrio(uvccCam *cam, uint8_t request, uint8_t *value);
/** Params and return as \ref uvccRWAutoFocus(). */
int uvccRWExposure(uvccCam *cam, uint8_t request, uint32_t *value);
/* TODO: relative exposure */
/**
 * uvccRequest wrapper function for auto focus, UVC_GET_CUR result can be found in \ref uvcc_auto_returns.
 * \param cam
 * \param request uvc request, valid values are those in \ref uvc_bRequest
 * \param value value to set or buf to receive value in
 * \return 0 on success != 0 on fail
 */
int uvccRWAutoFocus(uvccCam *cam, uint8_t request, uint8_t *value);
/** Params and return as \ref uvccRWAutoFocus(). */
int uvccRWFocus(uvccCam *cam, uint8_t request, uint16_t *value);
/* TODO: relative focus */
/** Params and return as \ref uvccRWAutoFocus(). */
int uvccRWIris(uvccCam *cam, uint8_t request, int16_t *value);
/** Params and return as \ref uvccRWAutoFocus(). */
/* TODO: relative iris */
int uvccRWBacklightCompensation(uvccCam *cam, uint8_t request, uint16_t *value);
/** Params and return as \ref uvccRWAutoFocus(). */
int uvccRWBrightness(uvccCam *cam, uint8_t request, int16_t *value);
/** Params and return as \ref uvccRWAutoFocus(). */
int uvccRWContrast(uvccCam *cam, uint8_t request, uint16_t *value);
/** Params and return as \ref uvccRWAutoFocus(). */
int uvccRWGain(uvccCam *cam, uint8_t request, uint16_t *value);
/** Params and return as \ref uvccRWAutoFocus(), UVC_GET_CUR result can be found in \ref uvcc_plf_returns. */
int uvccRWPowerLineFrequency(uvccCam *cam, uint8_t request, uint8_t *value);
/** Params and return as \ref uvccRWAutoFocus(), UVC_GET_CUR result can be found in \ref uvcc_auto_returns. */
int uvccRWAutoHue(uvccCam *cam, uint8_t request, uint8_t *value);
/** Params and return as \ref uvccRWAutoFocus(). */
int uvccRWHue(uvccCam *cam, uint8_t request, int16_t *value);
/** Params and return as \ref uvccRWAutoFocus(). */
int uvccRWSaturation(uvccCam *cam, uint8_t request, uint16_t *value);
/** Params and return as \ref uvccRWAutoFocus(). */
int uvccRWSharpness(uvccCam *cam, uint8_t request, uint16_t *value);
/** Params and return as \ref uvccRWAutoFocus(). */
int uvccRWGamma(uvccCam *cam, uint8_t request, uint16_t *value);
/** Params and return as \ref uvccRWAutoFocus(), UVC_GET_CUR result can be found in \ref uvcc_auto_returns. */
int uvccRWAutoWhiteBalanceTemp(uvccCam *cam, uint8_t request, uint8_t *value);
/** Params and return as \ref uvccRWAutoFocus(). */
int uvccRWWhiteBalanceTemp(uvccCam *cam, uint8_t request, uint16_t *value);
/** Params and return as \ref uvccRWAutoFocus(), UVC_GET_CUR result can be found in \ref uvcc_auto_returns. */
int uvccRWAutoWhiteBalanceComponent(uvccCam *cam, uint8_t request, uint8_t *value);
/**
 * uvccRequest wrapper function for white balance component, see table 4-39 in UVC spec, p.98 (111) for further info.
 * \param cam
 * \param request uvc request, valid values are those in \ref uvc_bRequest
 * \param blue blue component
 * \param red red component
 * \return 0 on success != 0 on fail
 */
int uvccRWWhiteBalanceComponent(uvccCam *cam, uint8_t request, uint16_t *blue, uint16_t *red);

/** @} */

/**
 * \mainpage Libuvcc description
 * \section intro Intro
 * Libuvcc in no way implements all of the uvc-standard. It's main purpose is help people control some of the basic functions available in their webcams.<br />
 *	This is very much an alpha-release, return values and function arguments may very well change in future releases.
 *
 * Bug reports are always welcome at sluggo@dm9.se
 *
 * \section usage Usage
 * \ref uvccInit() (normally when your program starts).
 *
 * Get a hold of your cam using one of the uvccGetCam... functions.
 *
 * Open the cam using \ref uvccOpenCam() when you want to send requests to it. Don't keep it open if you don't need it, opening it means you have exclusive access to the default control pipe on the video control interface.
 *
 * Make sure your cam support the request you want to send by calling \ref uvccSendInfoRequest() with the appropiet params and & the result with \ref uvcc_info_flags.
 *
 * If you're setting a numerical value (not on/off) get the min/max/res values and make sure your value is usable by <code>value<sub>n</sub> = n * (max - min) / res</code> where <code>min &le; value<sub>n</sub> &le; max</code>.
 *
 * Send your requests using one of the \ref reqs, \ref uvccSendRequest() or \ref uvccSendRawRequest().
 *
 * \ref uvccCloseCam() when you're done sending requests. You can open/close a cam as many times as you want.
 *
 * When you're done with your cam release it using \ref uvccReleaseCam().
 *
 * \ref uvccExit() (normally when your program exits).
 *
 * \section todo TODO
 *	- make sure all functions set last_error(_fn)
 *	- fix return values
 *	- as always.. more error handeling
 *  - fix handling of not responding devices
 *	- error-request, UVC spec p.79 (92)
 *	- more PU-requests
 *  - rewrite <code>uvcc_err</code> -> <code>log(int lvl, char *s, IOReturn r)</code> then switch lvl if !logger
 *	- option to only use external devices (ref: BusProper.m:251)
 *  - get dims @ fps
 *  - [see source]
 *
 * \section notes Additional notes
 * Using IOUSBDeviceInterface197 and IOUSBInterfaceInterface197 means IOUSBFamily 1.9.7 or above is required, that is os x >= 10.2.5. If anyone wants it's most certainly possible to rewrite for earlier versions.<br />
 * The code looks a bit incoherent, think of it as everything that is directly callable by a user is ObjC-formatted and the other stuff is more c. Just wanted to point out that i'm not a big fan of oop. not a big fan at all..<br />
 * Look at "Mac OS X 10.6 Core Library > Drivers, Kernel & Hardware > User-Space Device Access > USB Device Inteface Guide" (especially the SampleUSBMIDIDriver and Deva_Example projects), USBProberV2 ( http://opensource.apple.com/source/IOUSBFamily/IOUSBFamily-540.4.1/USBProberV2 ), Dominic Szablewski's uvc camera control ( http://www.phoboslab.org/log/2009/07/uvc-camera-control-for-mac-os-x ) and the libusb project ( http://libusb.org ) to see where this code came from..
 */

#endif
