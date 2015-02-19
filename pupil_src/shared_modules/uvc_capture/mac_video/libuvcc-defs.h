/* warning: mixed naming conventions! */
#ifndef __LIBUVCC_DEFS_H__
#define __LIBUVCC_DEFS_H__

#include <IOKit/usb/USB.h>

/** \defgroup flags Datastructures and definitions
 * The definitions and datastructures used by libuvcc.
 * @{ */

/** \defgroup defs Definitions
 * @{ */
/** \defgroup usbbasecls USB base classes
  * @{ */
#define USB_BASE_CLASS_DEV		0x00
#define USB_BASE_CLASS_COM		0x02
#define USB_BASE_CLASS_HID		0x03
#define USB_BASE_CLASS_PHYS		0x05
#define USB_BASE_CLASS_IMG		0x06
#define USB_BASE_CLASS_CDC		0x0a
#define USB_BASE_CLASS_VID		0x0e
#define USB_BASE_CLASS_MISC		0xef
#define USB_BASE_CLASS_APP		0xfe
#define USB_BASE_CLASS_VEND		0xff
/** @} */
/** \defgroup usbsubcls USB sub classes
  * @{ */
/** only subclass i'm interested in */
#define USB_SUB_CLASS_VID_CTRL	0x01
/** @} */
/** \defgroup bmrt USB bmRequestType bits
 * @{ */
#define USB_RT_TD_SET		0x00		/**< bit 7, transfer direction (out) */
#define USB_RT_TD_GET		0x80		/**< 10000000b (in) */
#define USB_RT_TY_STD		(0x00 << 5)	/**< bit 6,5, type */
#define USB_RT_TY_CLASS		(0x01 << 5)	/**< 00100000b */
#define USB_RT_TY_VENDOR	(0x02 << 5)	/**< 01000000b */
#define USB_RT_TY_RESERVED	(0x03 << 5)	/**< 01100000n */
#define USB_RT_RE_DEVICE	0x00		/**< bit 4-0, reciever device */
#define USB_RT_RE_IFACE		0x01		/**< reciever interface */
#define USB_RT_RE_ENDPOINT	0x02		/**< reciever endpoint */
#define USB_RT_RE_OTHER		0x03		/**< reciever other, reserved */
/** @} */
/** \defgroup usbstdreq USB standard requests
 * Standars requests, see usb2 spec table 9-4, p.251 (279). Values 0x02 and 0x04 are reserved for future use.
 * @{ */
#define USB_REQ_GET_STATUS	0x00
#define USB_REQ_CLEAR_FEAT	0x01
/* 0x02 reserved for future use */
#define USB_REQ_SET_FEAT	0x03
/* 0x04 reserved for future use */
#define USB_REQ_SET_ADDR	0x05
#define USB_REQ_GET_DESC	0x06
#define USB_REQ_SET_DESC	0x07
#define USB_REQ_GET_CONF	0x08
#define USB_REQ_SET_CONF	0x09
#define USB_REQ_GET_IFACE	0x0a
#define USB_REQ_SET_IFACE	0x0b
#define USB_REQ_SYNCH_FRAME	0x0c
/** @} */
/** \defgroup usbstdreq USB standard requests
 * Standard description types, see usb2 spec table 9-5, p.251-252.
 * @{ */
#define USB_DT_DEVICE			0x01
#define USB_DT_CONF				0x02
#define USB_DT_STRING			0x03
#define USB_DT_IFACE			0x04
#define USB_DT_ENDPOINT			0x05
#define USB_DT_DEV_QUALIFIER	0x06
#define USB_DT_OTHER_SPEED_CONF	0x07
#define USB_DT_IFACE_POWER		0x08
/** @} */
/** \defgroup uvc_bRequest UVC bRequest values
 * UVC bRequest values, see uvc spec section 4.1 p.73 (86).
 * @{ */
#define UVC_RC_UNDEFINED					0x00
#define UVC_SET_CUR							0x01
#define UVC_GET_CUR							0x81
#define UVC_GET_MIN							0x82
#define UVC_GET_MAX							0x83
#define UVC_GET_RES							0x84
#define UVC_GET_LEN							0x85
#define UVC_GET_INFO						0x86
#define UVC_GET_DEF							0x87
/** @} */
/** \defgroup uvc_wValues UVC wValue values
 * @{ */
#define UVC_VC_CONTROL_UNDEFINED				0x00
#define UVC_VC_VIDEO_POWER_MODE_CONTROL			0x01
#define UVC_VC_REQUEST_ERROR_CODE_CONTROL		0x02
#define UVC_CT_CONTROL_UNDEFINED				0x00
#define UVC_CT_SCANNING_MODE_CONTROL			(0x01 << 8)
#define UVC_CT_AE_MODE_CONTROL					(0x02 << 8)
#define UVC_CT_AE_PRIORITY_CONTROL				(0x03 << 8)
#define UVC_CT_EXPOSURE_TIME_ABSOLUTE_CONTROL	(0x04 << 8)
#define UVC_CT_EXPOSURE_TIME_RELATIVE_CONTROL	(0x05 << 8)
#define UVC_CT_FOCUS_ABSOLUTE_CONTROL			(0x06 << 8)
#define UVC_CT_FOCUS_RELATIVE_CONTROL			(0x07 << 8)
#define UVC_CT_FOCUS_AUTO_CONTROL				(0x08 << 8)
#define UVC_CT_IRIS_ABSOLUTE_CONTROL			(0x09 << 8)
#define UVC_CT_IRIS_RELATIVE_CONTROL			(0x0a << 8)
#define UVC_CT_ZOOM_ABSOLUTE_CONTROL			(0x0b << 8)
#define UVC_CT_ZOOM_RELATIVE_CONTROL			(0x0c << 8)
#define UVC_CT_PANTILT_ABSOLUTE_CONTROL			(0x0d << 8)
#define UVC_CT_PANTILT_RELATIVE_CONTROL			(0x0e << 8)
#define UVC_CT_ROLL_ABSOLUTE_CONTROL			(0x0f << 8)
#define UVC_CT_ROLL_RELATIVE_CONTROL			(0x10 << 8)
#define UVC_CT_PRIVACY_CONTROL					(0x11 << 8)
#define UVC_PU_CONTROL_UNDEFINED						(0x00 << 8)
#define UVC_PU_BACKLIGHT_COMPENSATION_CONTROL			(0x01 << 8)
#define UVC_PU_BRIGHTNESS_CONTROL						(0x02 << 8)
#define UVC_PU_CONTRAST_CONTROL							(0x03 << 8)
#define UVC_PU_GAIN_CONTROL								(0x04 << 8)
#define UVC_PU_POWER_LINE_FREQUENCY_CONTROL				(0x05 << 8)
#define UVC_PU_HUE_CONTROL								(0x06 << 8)
#define UVC_PU_SATURATION_CONTROL						(0x07 << 8)
#define UVC_PU_SHARPNESS_CONTROL						(0x08 << 8)
#define UVC_PU_GAMMA_CONTROL							(0x09 << 8)
#define UVC_PU_WHITE_BALANCE_TEMPERATURE_CONTROL		(0x0a << 8)
#define UVC_PU_WHITE_BALANCE_TEMPERATURE_AUTO_CONTROL	(0x0b << 8)
#define UVC_PU_WHITE_BALANCE_COMPONENT_CONTROL			(0x0c << 8)
#define UVC_PU_WHITE_BALANCE_COMPONENT_AUTO_CONTROL		(0x0d << 8)
#define UVC_PU_DIGITAL_MULTIPLIER_CONTROL				(0x0e << 8)
#define UVC_PU_DIGITAL_MULTIPLIER_LIMIT_CONTROL			(0x0f << 8)
#define UVC_PU_HUE_AUTO_CONTROL							(0x10 << 8)
#define UVC_PU_ANALOG_VIDEO_STANDARD_CONTROL			(0x11 << 8)
#define UVC_PU_ANALOG_LOCK_STATUS_CONTROL				(0x12 << 8)
/** @} */
/** \defgroup uvc_wIndex UVC wIndex upper bits
 * @{ */
#define UVC_CT_INDEX	(0x01 << 8)
#define UVC_PU_INDEX	(0x02 << 8)
/** @} */

/** \defgroup uvcc_defs Libuvcc definitions
 * Common bmRequestTypes and some other stuff. See section 4.2.1 and 4.2 in uvc spec.
 * @{ */
/** \defgroup uvcc_bmrt Common bmReqTypes
 * @{ */
#define UVCC_BMRT_SET	USB_RT_TD_SET | USB_RT_TY_CLASS | USB_RT_RE_IFACE
#define UVCC_BMRT_GET	USB_RT_TD_GET | USB_RT_TY_CLASS | USB_RT_RE_IFACE
/** @} */

/** \defgroup uvcc_info_flags Info flags
 * Flags to test result of \ref uvccSendInfoRequest().<br />Value data type: int8_t (bitmap), see table 4-3 in UVC spec, p.75 (88).
 * @{ */
#define UVCC_INFO_GET_SUPPORT           (int8_t)0x01
#define UVCC_INFO_SET_SUPPORT           (int8_t)0x02
#define UVCC_INFO_DISABLED_BY_AUTOCTRL  (int8_t)0x04
#define UVCC_INFO_ASYNC_CTRL            (int8_t)0x08
/** @} */

/** \defgroup uvcc_auto_returns Auto values
 * Possible return vales of auto-functions.<br />Value data type: uint8_t (boolean), see table 4-16 in UVC spec, p.85 (98).
 * @{ */
#define UVCC_AUTO_OFF   (uint8_t)0x00
#define UVCC_AUTO_ON    (uint8_t)0x01
/** @} */
/** \defgroup uvcc_scanning_mode_returns Scanning mode values
 * Possible results of \ref uvccRWScanningMode() and corresponding \ref uvccSendRequest().<br />Value data type: uint8_t (boolean), see table 4-9 in UVC spec, p.81 (94).
 * @{ */
#define UVCC_SCANNING_MODE_INTERLACED   (uint8_t)0x00
#define UVCC_SCANNING_MODE_PROGRESSIVE  (uint8_t)0x01
/** @} */
/** \defgroup uvcc_et_mode_returns Exposure mode values
 * Possible results of \ref uvccRWAutoExposureMode() and corresponding \ref uvccSendRequest().<br />Value data type: int8_t (bitmap), see table 4-10 in UVC spec, p.81-82 (94-95).
 * @{ */
#define UVCC_EXPOSURE_MODE_MANUAL           (int8_t)0x01
#define UVCC_EXPOSURE_MODE_AUTO             (int8_t)0x02
#define UVCC_EXPOSURE_MODE_SHUTTER_PRIO     (int8_t)0x04
#define UVCC_EXPOSURE_MODE_APERTURE_PRIO    (int8_t)0x08
/** @} */
/** \defgroup uvcc_et_prio_returns Exposure priority values
 * Possible results of \ref uvccRWAutoExposurePrio() and corresponding \ref uvccSendRequest().<br />Value data type: int8_t, see table 4-11 in UVC spec, p.82 (95).
 * @{ */
#define UVCC_EXPOSURE_PRIO_CTRL_CONSTANT    (int8_t)0x00
#define UVCC_EXPOSURE_PRIO_CTRL_DYNAMIC     (int8_t)0x01
/** @} */
/** \defgroup uvcc_plf_returns Power line frequency values
 * Possible results of \ref uvccRWPowerLineFrequency() and corresponding \ref uvccSendRequest().<br />Value data type: int8_t, see table 4-32 in UVC spec, p.95 (108).
 * @{ */
#define UVCC_POWER_LINE_FREQUENCE_DISABLED  (int8_t)0x00
#define UVCC_POWER_LINE_FREQUENCY_50HZ      (int8_t)0x01
#define UVCC_POWER_LINE_FREQUENCY_60HZ      (int8_t)0x02
/** @} */
/** @} */

/** @} */

/**
 * To shield you from the terrors of os x internals =). Note that 1.5 and 12 mbit/s are defined in both usb 1.0 and 1.1, hence UVCC_DEV_SPEED_USB_1A and UVCC_DEV_SPEED_USB_1B.
 */
enum uvccDevSpeed {
	UVCC_DEV_SPEED_1_5_MBPS		= kUSBDeviceSpeedLow,
	UVCC_DEV_SPEED_12_MBPS		= kUSBDeviceSpeedFull,
	UVCC_DEV_SPEED_480_MBPS		= kUSBDeviceSpeedHigh,
	UVCC_DEV_SPEED_5000_MBPS	= kUSBDeviceSpeedSuper,
	UVCC_DEV_SPEED_USB_1A		= UVCC_DEV_SPEED_1_5_MBPS,
	UVCC_DEV_SPEED_USB_1B		= UVCC_DEV_SPEED_12_MBPS,
	UVCC_DEV_SPEED_USB_2		= UVCC_DEV_SPEED_480_MBPS,
	UVCC_DEV_SPEED_USB_3		= UVCC_DEV_SPEED_5000_MBPS
};

/**
 * Errors that might be returned by some functions.
 */
enum uvccError
{
    UVCC_ERR_CREATE_MASTER_PORT_FAIL = -9,
	UVCC_ERR_CAM_IS_NULL,
	UVCC_ERR_INTERFACE_IS_NULL,
	UVCC_ERR_CTRL_REQUEST_IS_NULL,
	UVCC_ERR_UNKNOWN_REQUEST,
	UVCC_ERR_UNKNOWN_UVC_REQUEST,
	UVCC_ERR_USB_OPEN_FAILED,
    UVCC_ERR_CTRL_REQUEST_UNSUPPORTED_OR_MALFORMATTED,
	UVCC_ERR_CTRL_REQUEST_FAILED,
	UVCC_ERR_NO_ERROR = 0
};

/**
 * Warnings that might be returned from some functions.
 */
enum uvccWarning
{
    UVCC_WARN_NO_WARNING,
    UVCC_WARN_LOGGING_TO_STDERR
};

/**
 * Requests to send as argument to \ref uvccSendRequest() and \ref uvccSendInfoRequest().
 */
enum uvccRequest
{	/* note that these are just indicies in predef_reqs so don't change the order ! */
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
	UVCC_REQ_HUE_AUTO,
	UVCC_REQ_HUE_ABS,
	UVCC_REQ_SATURATION_ABS,
	UVCC_REQ_SHARPNESS_ABS,
    UVCC_REQ_GAMMA_ABS,
    UVCC_REQ_WB_TEMP_AUTO,
	UVCC_REQ_WB_TEMP_ABS,
	UVCC_REQ_WB_COMPONENT_AUTO,
	UVCC_REQ_WB_COMPONENT_ABS,
	__UVCC_REQ_OUT_OF_RANGE
};

/**
 * Vendor id and product id.
 */
struct uvccModelID
{
	uint16_t idVendor;
	uint16_t idProduct;
};

/**
 * Device descriptor, differs a bit from IOUSBDeviceDescriptor. See source, internal function get_dev_desc for further information.
 */
struct uvccDevDesc
{	/* "soft" info */
	uint8_t		bDevClass;
	uint8_t		bDevSubClass;
	uint8_t		bDevProtocol;
	uint8_t		bDevSpeed;
	/* "hard" info */
	uint16_t	hwIdVendor;
	uint16_t	hwIdProduct;
	uint16_t	hwRelNo;
	uint8_t		bNumConfs;
	/* string indicies */
	uint8_t		bIManufacturer;
	uint8_t		bIProduct;
	uint8_t		bISerialNumber;
};

/**
 * uvccCam, the guts!
 */
struct _uvccCam
{
	uint32_t idLocation;
	/* mId is actually just a pointer into the devDesc */
	struct uvccModelID *mID;
	//IOUSBDeviceDescriptor devDesc;
	IOUSBDeviceInterface197 **devIf;
	struct uvccDevDesc devDesc;
	IOUSBInterfaceInterface197 **ctrlIf;
    uint8_t ifNo;
	/* os x registry name of cam, can be cast straight to NSString* */
	CFStringRef regName;
};
/**
 * For your convenience.
 */
typedef struct _uvccCam uvccCam;


/** @} */

#endif
