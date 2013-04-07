/* to use uvcc from source you need to import the IOKit framework */
#ifndef __UVCC_H__
#define __UVCC_H__

#include <CoreFoundation/CoreFoundation.h>
#include <IOKit/usb/IOUSBLib.h>

#include "uvcc-defs.h"

enum uvccError
{
    UVCCE_CREATE_MASTER_PORT_FAIL = -9,
	UVCCE_CAM_IS_NULL,
	UVCCE_INTERFACE_IS_NULL,
	UVCCE_CTRL_REQUEST_IS_NULL,
	UVCCE_UNKNOWN_REQUEST,
	UVCCE_UNKNOWN_UVC_REQUEST,
	UVCCE_USB_OPEN_FAILED,
    UVCCE_CTRL_REQUEST_UNSUPPORTED_OR_MALFORMATTED,   /* happens when unsupported req is sent  */
	UVCCE_CTRL_REQUEST_FAILED,
	UVCCE_NO_ERROR = 0
};
enum uvccWarning
{
    UVCCW_NO_WARNING,
    UVCCW_LOGGING_TO_STDERR
};
/* theese are just indexes in predef_reqs.. so don't change the order! */
enum uvccRequests
{
    UVCC_REQ_SCANNING_MODE,
    UVCC_REQ_EXPOSURE_AUTOMODE,
    UVCC_REQ_EXPOSURE_AUTOPRIO,
	UVCC_REQ_EXPOSURE_ABS,
	UVCC_REQ_EXPOSURE_REL,
	UVCC_REQ_FOCUS_AUTO,
	UVCC_REQ_FOCUS_ABS,
	UVCC_REQ_FOCUS_REL,
	UVCC_REQ_IRIS_ABS,
	UVCC_REQ_IRIS_REL,
    UVCC_REQ_ZOOM_ABS,
    UVCC_REQ_BACKLIGHT_COMPENSATION_ABS,
	UVCC_REQ_BRIGHTNESS_ABS,
	UVCC_REQ_CONTRAST_ABS,
	UVCC_REQ_GAIN_ABS,
    UVCC_REQ_POWER_LINE_FREQ,
	UVCC_REQ_SATURATION_ABS,
	UVCC_REQ_SHARPNESS_ABS,
    UVCC_REQ_GAMMA_ABS,
    UVCC_REQ_WB_TEMP_AUTO,
	UVCC_REQ_WB_TEMP_ABS,
	__UVCC_REQ_OUT_OF_RANGE
};

struct uvccModelId
{
	UInt16 idVendor;
	UInt16 idProduct;
};

/* just a wrapper to make it easier to use with QTKit */
struct uvccCam
{
	UInt32 idLocation;
	/* mId is actually just a pointer into the devDesc */
	struct uvccModelId *mId;
	IOUSBDeviceDescriptor devDesc;
	IOUSBDeviceInterface197 **devIf;
	IOUSBInterfaceInterface197 **ctrlIf;
    UInt8 ifNo;
};

/**
 * inits uvcc..
 * @return 0 on success
 */
int uvccInit();

/**
 * exits uvcc
 */
void uvccExit();

/**
 * populate cams with all connected devices with video control interfaces
 * @param cams to hold list, needs to be freed by freeCamList
 * @return length of list or -1
 */
int uvccGetCamList(struct uvccCam ***list);

/**
 * release and free list of cams
 * @param cams list of cams
 * @param len length of list
 */
void uvccReleaseCamList(struct uvccCam **list,
					 int len);

/**
 * for easy incorporaration with qtkit-projects
 * @param uniqueID of QTCaptureDevice
 * @param cam struct to fill
 * @return 0 on success != 0 otherwise
 */
int uvccGetCamWithQTUniqueID(char *uID, struct uvccCam *cam);

/**
 * get (first) cam identified by modelId
 * @param mId modelId identifying cam
 * @param cam struct to fill
 * @return 0 on success != 0 otherwise
 */
int uvccGetCamWithModelId(struct uvccModelId *mId, struct uvccCam *cam);

/**
 * Release single cam
 * @param cam to release
 */
void uvccReleaseCam(struct uvccCam *cam);

/**
 * helper function to be able to compare uvccCams with QTCaptureDevices using the latters [QTCaptureDevice uniqueID].
 * @param cam
 * @param buf to store uniqueID in
 * @return pointer to uniqueID or NULL
 */
char *uvccCamQTUniqueID(struct uvccCam *cam,
						char buf[19]);

/**
 * gets the manufacturer/product/serial (any/all may be empty strings)
 * @param cam
 * @param buf[128] to store string in
 * @return length of string (in UniChars == bytelength/2) or < 0 on error
 */
int uvccCamManufacturer(struct uvccCam *cam,
						UniChar buf[128]);
int uvccCamProduct(struct uvccCam *cam,
					UniChar buf[128]);
int uvccCamSerialNumber(struct uvccCam *cam,
						UniChar buf[128]);

/**
 * convert from UniChar to char
 * @param src, source
 * @param dst, destination
 * @param len, number of UniChars to convert
 * @param errChar, char to set if src[i] is non-ascii or 0 to skip them
 * @return length of destination
 */
int uvccUni2Char(UniChar *src,
				 char *dst,
				 int len,
				 int errChar);

/**
 * send predefined uvcc request to cam
 * @param cam
 * @param bRequest uvc request to send, valid values are UVC_SET_CUR, UVC_GET_CUR, UVC_GET_MIN, UVC_GET_MAX, UVC_GET_RES, UVC_GET_LEN or UVC_GET_DEF
 * @param uvccRequest uvcc request to send, valid values are those in uvccRequests
 * @param pData data to send or buf to receive data in
 * @return 0 on success or error code
 */
unsigned int uvccSendRequest(struct uvccCam *cam,
							 UInt8 bRequest,
							 unsigned int uvccRequest,
							 void *pData);
/**
 * for the brave.. send a raw request!
 * @param cam
 * @param request to send
 * @return 0 on success or error code
 */
unsigned int uvccSendRawRequest(struct uvccCam *cam,
								IOUSBDevRequest *request);

/**
 * wrapper for the UVC_GET_INFO
 * @param cam
 * @param uvccRequest to get info about
 * @return bitmap to be &'d with UVCC_INFO_* on success or < 0 on fail.
 */
int8_t uvccRequestInfo(struct uvccCam *cam,
                       unsigned int uvccRequest);
/**
 * uvccRequest wrapper functions..
 * @oaram cam
 * @param request uvc request, valid values are UVC_SET_CUR, UVC_GET_CUR, UVC_GET_MIN, UVC_GET_MAX, UVC_GET_RES, UVC_GET_LEN or UVC_GET_DEF
 * @param value if request == UVC_SET_CUR (function name) will be set to this, otherwise ignored.
 * @return on success requested value for UVC_GET_*, [@ref value] for UVC_SET_CUR, < 0 on fail.
 */
int8_t uvccScanningMode(struct uvccCam *cam, UInt8 request, int8_t value);
int8_t uvccExposureMode(struct uvccCam *cam, UInt8 request, int8_t value);
int8_t uvccExposurePrio(struct uvccCam *cam, UInt8 request, int8_t value);
int32_t uvccExposure(struct uvccCam *cam, UInt8 request, int32_t value);
/* TODO: relative exposure */
int8_t uvccAutoFocus(struct uvccCam *cam, UInt8 request, int8_t value);
int16_t uvccFocus(struct uvccCam *cam, UInt8 request, int16_t value);
/* TODO: relative focus */
int16_t uvccIris(struct uvccCam *cam, UInt8 request, int16_t value);
/* TODO: relative iris */
int16_t uvccBacklightCompensation(struct uvccCam *cam, UInt8 request, int16_t value);
int16_t uvccBrightness(struct uvccCam *cam, UInt8 request, int16_t value);
int16_t uvccContrast(struct uvccCam *cam, UInt8 request, int16_t value);
int16_t uvccGain(struct uvccCam *cam, UInt8 request, int16_t value);
int8_t uvccPowerLineFrequency(struct uvccCam *cam, UInt8 request, int8_t value);
int16_t uvccSaturation(struct uvccCam *cam, UInt8 request, int16_t value);
int16_t uvccSharpness(struct uvccCam *cam, UInt8 request, int16_t value);
int16_t uvccGamma(struct uvccCam *cam, UInt8 request, int16_t value);
int8_t uvccAutoWhiteBalanceTemp(struct uvccCam *cam, UInt8 request, int8_t value);
int16_t uvccWhiteBalanceTemp(struct uvccCam *cam, UInt8 request, int16_t value);

#endif
