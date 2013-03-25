/* warning: mixed naming conventions! */

/* usb base classes */
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
/* only subclass i'm interested in */
#define USB_SUB_CLASS_VID_CTRL	0x01
/* bmRequestType crap */
#define USB_RT_TD_SET		0x00		/* bit 7, transf. dir */
#define USB_RT_TD_GET		0x80		/* 10000000 */
#define USB_RT_TY_STD		(0x00 << 5)	/* bit 6,5, type */
#define USB_RT_TY_CLASS		(0x01 << 5)	/* 00100000 */
#define USB_RT_TY_VENDOR	(0x02 << 5)	/* 01000000 */
#define USB_RT_TY_RESERVED	(0x03 << 5)	/* 01100000 */
#define USB_RT_RE_DEVICE	0x00		/* bit 4-0, reciever */
#define USB_RT_RE_IFACE		0x01		/* interface */
#define USB_RT_RE_ENDPOINT	0x02
#define USB_RT_RE_OTHER		0x03		/* rest is reserved */
/* std requests, see usb2 spec table 9-4, p.251 (279) */
#define USB_REQ_GET_STATUS	0x00
#define USB_REQ_CLEAR_FEAT	0x01
#define USB_REQ_SET_FEAT	0x03
#define USB_REQ_SET_ADDR	0x05
#define USB_REQ_GET_DESC	0x06
#define USB_REQ_SET_DESC	0x07
#define USB_REQ_GET_CONF	0x08
#define USB_REQ_SET_CONF	0x09
#define USB_REQ_GET_IFACE	0x0a
#define USB_REQ_SET_IFACE	0x0b
#define USB_REQ_SYNCH_FRAME	0x0c
/* std desc types, see usb2 spec table 9-5, p.251-252 */
#define USB_DT_DEVICE			0x01
#define USB_DT_CONF				0x02
#define USB_DT_STRING			0x03
#define USB_DT_IFACE			0x04
#define USB_DT_ENDPOINT			0x05
#define USB_DT_DEV_QUALIFIER	0x06
#define USB_DT_OTHER_SPEED_CONF	0x07
#define USB_DT_IFACE_POWER		0x08
/* wIndex tjosan hejsan */
#define UVC_CT_INDEX	(0x01 << 8)
#define UVC_PU_INDEX	(0x02 << 8)
/* uvc requests, see uvc spec section 4.1 p.73 (86) */
#define UVC_RC_UNDEFINED					0x00
#define UVC_SET_CUR							0x01
#define UVC_GET_CUR							0x81
#define UVC_GET_MIN							0x82
#define UVC_GET_MAX							0x83
#define UVC_GET_RES							0x84
#define UVC_GET_LEN							0x85
#define UVC_GET_INFO						0x86
#define UVC_GET_DEF							0x87
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
/* white balance is the only auto feature in the pu i'm using right now */
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
/* the most common bmReqTypes */
#define UVCC_BMRT_SET	USB_RT_TD_SET | USB_RT_TY_CLASS | USB_RT_RE_IFACE
#define UVCC_BMRT_GET	USB_RT_TD_GET | USB_RT_TY_CLASS | USB_RT_RE_IFACE
/* here are some returns.. see section 4.2.1 and 4.2 in uvc spec */
#define UVCC_INFO_GET_SUPPORT           0x01
#define UVCC_INFO_SET_SUPPORT           0x02
#define UVCC_INFO_DISABLED_BY_AUTOCTRL  0x04
#define UVCC_INFO_ASYNC_CTRL            0x08

#define UVCC_AUTO_OFF   0x00
#define UVCC_AUTO_ON    0x01

#define UVCC_SCANNING_MODE_INTERLACED   0x00
#define UVCC_SCANNING_MODE_PROGRESSIVE  0x01

#define UVCC_EXPOSURE_MODE_MANUAL           0x01
#define UVCC_EXPOSURE_MODE_AUTO             0x02
#define UVCC_EXPOSURE_MODE_SHUTTER_PRIO     0x04
#define UVCC_EXPOSURE_MODE_APERTURE_PRIO    0x08
#define UVCC_EXPOSURE_PRIO_CTRL_CONSTANT    0x00
#define UVCC_EXPOSURE_PRIO_CTRL_DYNAMIC     0x01

#define UVCC_POWER_LINE_FREQUENCE_DISABLED  0x00
#define UVCC_POWER_LINE_FREQUENCY_50HZ      0x01
#define UVCC_POWER_LINE_FREQUENCY_60HZ      0x02
