
/* by sluggo, sluggo@dm9.se ( http://dm9.se )
 *
 * TODO:
 *  - more PU-requests
 *  - rewrite uvcc_err -> log(lvl, s, r) then switch lvl if !logger
 *	- fix return values!
 *	- add char *uvccRequestErrorStr using
 *	  typedef IOReturn uvccReturn;
 *	  #define UVCC_RETURN_SUCCESS	kIOReturnSuccess
 *  - fix handling of not responding devices
 *  - listener (see USBProber)
 *  - get dims @ fps
 *  - [see source]
 */

#include <IOKit/IOKitLib.h>
#include <IOKit/IOCFPlugIn.h>
#include <mach/mach_init.h>
#include <mach/mach_port.h>
#include <asl.h>
#include <math.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "libuvcc.h"

#define UVCC_INIT_ERROR_MSG "You have to call uvccInit first dummy!\n"
#define UVCC_VERSION		"0.33a"

/* predefined requests */
static const IOUSBDevRequest predef_reqs[] =
{	/* bmReqType, bReq, wValue, wIndex, wLen, pData, wLenDone */
    { 0,0, UVC_CT_SCANNING_MODE_CONTROL, UVC_CT_INDEX, 0x01, NULL, 0},
    { 0,0, UVC_CT_AE_MODE_CONTROL, UVC_CT_INDEX, 0x01, NULL, 0 },
    { 0,0, UVC_CT_AE_PRIORITY_CONTROL, UVC_CT_INDEX, 0x01, NULL, 0 },
	{ 0,0, UVC_CT_EXPOSURE_TIME_ABSOLUTE_CONTROL, UVC_CT_INDEX, 0x04, NULL, 0 },
    /* 0x00: def, 0x01: inc, 0xff: dec */
	{ 0,0, UVC_CT_EXPOSURE_TIME_RELATIVE_CONTROL, UVC_CT_INDEX, 0x01, NULL, 0 },
	{ 0,0, UVC_CT_FOCUS_AUTO_CONTROL, UVC_CT_INDEX, 0x01, NULL, 0, },
	{ 0,0, UVC_CT_FOCUS_ABSOLUTE_CONTROL, UVC_CT_INDEX, 0x02, NULL, 0 },
	{ 0,0, UVC_CT_FOCUS_RELATIVE_CONTROL, UVC_CT_INDEX, 0x02, NULL, 0 },
	{ 0,0, UVC_CT_IRIS_ABSOLUTE_CONTROL, UVC_CT_INDEX, 0x02, NULL, 0 },
	{ 0,0, UVC_CT_IRIS_RELATIVE_CONTROL, UVC_CT_INDEX, 0x01, NULL, 0 },
	{ 0,0, UVC_CT_ZOOM_ABSOLUTE_CONTROL, UVC_CT_INDEX, 0x02, NULL, 0 },
    /* TODO: UVC_CT_ZOOM_RELATIVE_CONTROL */
    /* TODO: UVC_CT_PANTILT_ABSOLUTE_CONTROL */
    /* TODO: UVC_CT_PANTILT_RELATIVE_CONTROL */
    /* TODO: UVC_CT_ROLL_ABSOLUTE_CONTROL */
    /* TODO: UVC_CT_ROLL_RELATIVE_CONTROL */
    /* TODO: UVC_CT_PRIVACY_CONTROL */
    /* TODO: SU-shit.. */
    { 0,0, UVC_PU_BACKLIGHT_COMPENSATION_CONTROL, UVC_PU_INDEX, 0x02, NULL, 0 },
    /* fix brightness! it's a fucking signed value! */
    { 0,0, UVC_PU_BRIGHTNESS_CONTROL, UVC_PU_INDEX, 0x02, NULL, 0 },
    { 0,0, UVC_PU_CONTRAST_CONTROL, UVC_PU_INDEX, 0x02, NULL, 0 },
	{ 0,0, UVC_PU_GAIN_CONTROL, UVC_PU_INDEX, 0x02, NULL, 0 },
	{ 0,0, UVC_PU_POWER_LINE_FREQUENCY_CONTROL, UVC_PU_INDEX, 0x01, NULL, 0 },
	{ 0,0, UVC_PU_HUE_AUTO_CONTROL, UVC_PU_INDEX, 0x01, NULL, 0 },
	{ 0,0, UVC_PU_HUE_CONTROL, UVC_PU_INDEX, 0x02, NULL, 0 },
	{ 0,0, UVC_PU_SATURATION_CONTROL, UVC_PU_INDEX, 0x02, NULL, 0 },
	{ 0,0, UVC_PU_SHARPNESS_CONTROL, UVC_PU_INDEX, 0x02, NULL, 0 },
    { 0,0, UVC_PU_GAMMA_CONTROL, UVC_PU_INDEX, 0x02, NULL, 0 },
	{ 0,0, UVC_PU_WHITE_BALANCE_TEMPERATURE_AUTO_CONTROL, UVC_PU_INDEX, 0x01, NULL, 0 },
	{ 0,0, UVC_PU_WHITE_BALANCE_TEMPERATURE_CONTROL, UVC_PU_INDEX, 0x02, NULL, 0 },
	{ 0,0, UVC_PU_WHITE_BALANCE_COMPONENT_AUTO_CONTROL, UVC_PU_INDEX, 0x01, NULL, 0 },
	{ 0,0, UVC_PU_WHITE_BALANCE_COMPONENT_CONTROL, UVC_PU_INDEX, 0x04, NULL, 0 }
    /* TODO: UVC_PU_DIGITAL_MULTIPLIER_CONTROL */
    /* TODO: UVC_PU_DIGITAL_MULTIPLIER_LIMIT_CONTROL */
    /* TODO: UVC_PU_ANALOG_VIDEO_STANDARD_CONTROL (get) */
    /* TODO: UVC_PU_ANALOG_LOCK_STATUS_CONTROL (get) */
};

static IOUSBFindInterfaceRequest vid_ctrl_if_req =
{
	USB_BASE_CLASS_VID,
	USB_SUB_CLASS_VID_CTRL,
	kIOUSBFindInterfaceDontCare,
	kIOUSBFindInterfaceDontCare
};

/* see table 9-15/16, p.274 in usb2 spec */
typedef struct
{
	uint8_t		bLength;
	uint8_t		bDescriptorType;
	uint16_t	bString[127];
} usb_str;
typedef usb_str langid_arr;

/* we'll need our own iokit port otherwise everythings goes to shit when trying to communicate with more than one cam */
static mach_port_t	uvcc_port = 0;
/* the logger */
static aslclient logger;
/* the error-holder */
static IOReturn last_error;
static char *last_error_fn;

/* here are some internal funcs, listed at the bottom */
static CFDictionaryRef get_usb_service_dic();
static int get_cam_list(io_iterator_t devIter, uvccCam ***list);
static int fill_cam_struct(IOUSBDeviceInterface197 **devIf, uvccCam *cam);
static void set_cam_reg_name(io_service_t srv, uvccCam *cam);
static IOUSBInterfaceInterface197 **get_ctrl_if(IOUSBDeviceInterface197 **devIf);
static int get_dev_desc(IOUSBDeviceInterface197 **devIf, struct uvccDevDesc *dd);
						//IOUSBDeviceDescriptor *dd);
static int get_ctrl_ep(IOUSBDeviceInterface197 **devIf);
static int get_string_desc(IOUSBDeviceInterface197 **devIf, uint8_t index, UniChar buf[128]);
static void uvcc_err(const char *f, int r);
static const char *kio_return_str(int r);

/* funcs are listed in the same order as in the header */
int uvccInit()
{
    int ret = UVCC_ERR_NO_ERROR;
    /* mach_port_null -> default port to communicate on */
	kern_return_t kr = IOMasterPort(MACH_PORT_NULL, &uvcc_port);
	if(kr != kIOReturnSuccess || !uvcc_port)
	{
		uvcc_err("uvccInit: IOMasterPort", kr);
		return UVCC_ERR_CREATE_MASTER_PORT_FAIL;
	}
	logger = asl_open("se.dm9.uvcc", "uvcc logger facility", ASL_OPT_STDERR | ASL_OPT_NO_DELAY);
    if(!logger) ret = UVCC_WARN_LOGGING_TO_STDERR;
	last_error = kIOReturnSuccess;
	last_error_fn = "uvccInit";
	return ret;
}

void uvccExit()
{
	if(uvcc_port) mach_port_deallocate(mach_task_self(), uvcc_port);
    if(logger) asl_close(logger);
}

CFStringRef uvccVersion()
{
	return CFSTR(UVCC_VERSION);
}

int uvccGetCamList(uvccCam ***list)
{
	CFDictionaryRef	mRef;
	io_iterator_t devIter;
	kern_return_t kr;
	int nCams;
	/* make sure we've been initialized */
	if(!uvcc_port)
	{
		fprintf(stderr, UVCC_INIT_ERROR_MSG);
		return -1;
	}
	/* get the matching dic, get_usb_service_dic dumps error if needed */
	if(!(mRef = get_usb_service_dic())) return -1;
	/* get ALL the cams! */
	kr = IOServiceGetMatchingServices(uvcc_port, mRef, &devIter);
	if(kr != kIOReturnSuccess || !IOIteratorIsValid(devIter))
	{
		uvcc_err("uvccGetCamList: IOServiceGetMatchingServices", kr);
		return -1;
	}
	/* normally one ref of mdRef should be consumed by IOServiceGet... but a
	   bug in os x < Tiger can cause that to be missed. unfortunately you
	   can't get the retain count on a completely released obj in a safe manor
	   since they may or may not be free'd.
	if(CFGetRetainCount(mRef) > 0) CFRelease(mRef); */
	nCams = get_cam_list(devIter, list);
	IOObjectRelease(devIter);
	return nCams;
}

int uvccGetCamWithQTUniqueID(CFStringRef uId, uvccCam **cam)
{
	char cUId[19], idVendor[5], idProduct[5], location[9];
	struct uvccModelID mID;
	int nCams, i = 0;
	uvccCam **list;
	uint32_t sLoc;
	/* init cam */
	(*cam) = NULL;
	/* convert to c string and parse values */
	CFStringGetCString(uId, cUId, sizeof(cUId), kCFStringEncodingASCII);
	if(strlen(cUId) != 18 || cUId[0] != '0' || cUId[1] != 'x')
	{
		if(!logger) fprintf(stderr, "uvcc error! uvccGetCamWithQTUniqueID: supplied string is not an QTKit unique ID\n");
        else asl_log(logger, NULL, ASL_LEVEL_ERR, "uvccGetCamWithQTUniqueID: supplied string is not an QTKit unique ID");
		return -1;
	}
	/* copy and add ending */
	strncpy(location, &cUId[2], 8);
	location[8] = 0;
	strncpy(idVendor, &cUId[10], 4);
	idVendor[4] = 0;
	strncpy(idProduct, &cUId[14], 4);
	idProduct[4] = 0;
	sLoc = (uint32_t)strtol(location, NULL, 16);
	mID.idVendor = (uint16_t)strtol(idVendor, NULL, 16);
	mID.idProduct = (uint16_t)strtol(idProduct, NULL, 16);
	/* unfortunately you can't set location as matching dic value (since it's
	   not a cam prop).. */
	nCams = uvccGetCamsWithModelID(&mID, &list);
	if(nCams > 0)
	{	/* find matching cam and copy it into */
		for(i = 0; i < nCams; i++)
		{
			if(sLoc == list[i]->idLocation)
			{	/* we found it! */
				(*cam) = malloc(sizeof(uvccCam));
				if(!(*cam))
				{
					if(!logger) perror("uvcc error! uvccGetCamWithQTUniqueID: could not allocate memory for cam struct");
					else asl_log(logger, NULL, ASL_LEVEL_ERR, " uvccGetCamWithQTUniqueID: could not allocate memory for cam struct: %s", strerror(errno));
				}
				else memcpy((*cam), list[i], sizeof(uvccCam));
			}
			else
			{
				uvccReleaseCam(list[i]);
				free(list[i]);
			}
		}
		free(list);
	}
	if(!(*cam)) {
		if(!logger) fprintf(stderr, "uvcc error! uvccGetCamWithQTUniqueID: no camera with supplied QTKit unique ID was found\n");
        else asl_log(logger, NULL, ASL_LEVEL_ERR, "uvccGetCamWithQTUniqueID: no camera with supplied QTKit unique ID was found");
		return -1;
	}
	return 0;
}

int uvccGetCamsWithModelID(struct uvccModelID *mID, uvccCam ***list)
{
	CFDictionaryRef mRef;
	CFMutableDictionaryRef mdRef;
	CFNumberRef nRef;
	kern_return_t kr;
	io_iterator_t devIter;
	int nCams;

	if(!uvcc_port)
	{
		fprintf(stderr, UVCC_INIT_ERROR_MSG);
		return -1;
	}
	if(!(mRef = get_usb_service_dic())) return -1;
	/* set up the matching criteria for the device service we want */
	mdRef = CFDictionaryCreateMutableCopy(kCFAllocatorDefault, 0, mRef);
	CFRelease(mRef);
	if(!mdRef) {
		if(!logger) perror("uvcc error! uvccGetCamsWithModelId: CFDictionaryCreateMutableCopy returned NULL, could not create mutable dictionary.");
        else asl_log(logger, NULL, ASL_LEVEL_ERR, "uvccGetCamsWithModelId: CFDictionaryCreateMutableCopy returned NULL, could not create mutable dictionary.");
		return -1;
	}
	/* set required fields */
	nRef = CFNumberCreate(kCFAllocatorDefault, kCFNumberSInt16Type, &(mID->idVendor));
	CFDictionarySetValue(mdRef, CFSTR(kUSBVendorID), nRef);
	CFRelease(nRef);
	nRef = CFNumberCreate(kCFAllocatorDefault, kCFNumberSInt16Type, &(mID->idProduct));
	CFDictionarySetValue(mdRef, CFSTR(kUSBProductID), nRef);
	CFRelease(nRef);
	/* get the actual services */
	kr = IOServiceGetMatchingServices(uvcc_port, mdRef, &devIter);
	/*if(CFGetRetainCount(mdRef) > 0) CFRelease(mdRef);*/
	if(kr != kIOReturnSuccess || !IOIteratorIsValid(devIter))
	{
		uvcc_err("uvccGetCamsWithModelID: IOServiceGetMatchingServices", kr);
		return -1;
	}
	nCams = get_cam_list(devIter, list);
	IOObjectRelease(devIter);
	return nCams;
	/* this is just gettting the first one!
	 CFNumberRef nRef;
	 io_iterator_t devIter;
	 io_service_t camSrv;
	 IOUSBDeviceInterface197 **devIf;
	 IOCFPlugInInterface **pIf;
	 kern_return_t kr;
	 int32_t score;
	 HRESULT qiRes;
	 CFDictionaryRef mRef;
	 CFMutableDictionaryRef mdRef;
	...
	 * get the actual service *
	camSrv = IOServiceGetMatchingServices(uvcc_port, mdRef, &devIter);
	CFRelease(mdRef);
	* and these darn plugin interfaces *
	pIfKr = IOCreatePlugInInterfaceForService(camSrv, kIOUSBDeviceUserClientTypeID, kIOCFPlugInInterfaceID, &pIf, &score);
	if(pIfKr != kIOReturnSuccess || !pIf)
	{
		uvcc_err("uvccGetCamWithModelId: IOCreatePlugInInterfaceForService", pIfKr);
		return -1;
	}
	qiRes = (*pIf)->QueryInterface(pIf, CFUUIDGetUUIDBytes(kIOUSBDeviceInterfaceID), (LPVOID *)&devIf);
	IODestroyPlugInInterface(pIf);
	if(qiRes || !devIf)
	{
		if(!logger) perror("uvcc error! uvccGetCamWithModelId: QueryInterface failed");
        else asl_log(logger, NULL, ASL_LEVEL_ERR, "uvccGetCamWithModelId: QueryInterface failed: %s", strerror(errno));
		return -1;
	}
	* fill and return *
	if((fill_cam_struct(devIf, cam)) != 0)
	{
		(*devIf)->Release(devIf);
		return -1;
	}
	* get the registry name *
	set_cam_reg_name(camSrv, cam);
	 IOObjectRelease(camSrv);
	 return 0;
	 */
}

void uvccReleaseCamList(uvccCam **list, int len)
{
	int i;
	for(i = 0; i < len; i++)
	{
		uvccReleaseCam(list[i]);
		/* in lists each element is allocated */
		free(list[i]);
	}
	free(list);
}

void uvccReleaseCam(uvccCam *cam)
{
	(*(cam->devIf))->Release(cam->devIf);
    (*(cam->ctrlIf))->Release(cam->ctrlIf);
	CFRelease(cam->regName);
}

CFStringRef uvccCamQTUniqueID(uvccCam *cam)
{
	char buf[19];
	#if __LP64__
		sprintf(buf, "0x%08x%04x%04x",
				cam->idLocation,
				cam->mID->idVendor,
				cam->mID->idProduct);
	#else
		sprintf(buf, "0x%08lx%04x%04x",
				cam->idLocation,
				cam->mID->idVendor,
				cam->mID->idProduct);
	#endif
	return CFStringCreateWithCString(kCFAllocatorDefault, buf, kCFStringEncodingASCII);
	// printf("length %i", CFStringGetLength(ptr));
	// CFShowStr(ptr);
	// CFShow(ptr);
	// return ptr;
	// return CFStringCreateCopy(kCFAllocatorDefault,ptr);

}



int uvccUni2Char(UniChar *src, char *dst, int len, int errChar)
{
	int si, di;
	uint8_t *_src = (uint8_t *)src;
	for(si = 0, di = 0; si < 2*len; si += 2, di++)
	{	/* others will cause trouble.. */
		if(_src[si] < 32 || _src[si] > 126)
		{
			if(errChar > 0) dst[di] = errChar;
			else di--;
		}
		else dst[di] = _src[si];

	}
	dst[di] = 0;
	return di;
}

CFStringRef uvccCamManufacturer(uvccCam *cam)
{
	UniChar buf[128];
	int nchars = get_string_desc(cam->devIf, cam->devDesc.bIManufacturer, buf);
		if(nchars >0){
	CFStringRef str =  CFStringCreateWithCharacters(kCFAllocatorDefault, buf, nchars);
	return str;
	}
	else{return 0;
	}
}

CFStringRef uvccCamProduct(uvccCam *cam)
{
	UniChar buf[128];
	int nchars = get_string_desc(cam->devIf, cam->devDesc.bIProduct, buf);
	if(nchars >0){
	CFStringRef str =  CFStringCreateWithCharacters(kCFAllocatorDefault, buf, nchars);
	return str;
	}
	else{return 0;
	}
}
CFStringRef uvccCamSerialNumber(uvccCam *cam)
{
	UniChar buf[128];
	int nchars = get_string_desc(cam->devIf, cam->devDesc.bISerialNumber, buf);
	if(nchars >0){
	CFStringRef str =  CFStringCreateWithCharacters(kCFAllocatorDefault, buf, nchars);
	return str;
	}
	else{return 0;
	}
}

/*
 Lägg in resolution request!!
 */

int uvccOpenCam(uvccCam *cam) {
	IOReturn ior;
	if((ior = (*(cam->ctrlIf))->USBInterfaceOpenSeize(cam->ctrlIf)) != kIOReturnSuccess) {
		uvcc_err("uvccOpenCam: USBInterfaceOpenSeize", ior);
		last_error = ior;
		last_error_fn = "uvccOpenCam";
		return -1;
	}
	return 0;
}

void uvccCloseCam(uvccCam *cam) {
	(*(cam->ctrlIf))->USBInterfaceClose(cam->ctrlIf);
}

int uvccSendRequest(uvccCam *cam,
					uint8_t bRequest,
					enum uvccRequest uvccReq,
					void *pData)
{
	IOUSBDevRequest	req;
	uint8_t			bmRT;
	if(cam == NULL) return UVCC_ERR_CAM_IS_NULL;
	else if(cam->ctrlIf == NULL) return UVCC_ERR_INTERFACE_IS_NULL;
	if(uvccReq >= __UVCC_REQ_OUT_OF_RANGE) return UVCC_ERR_UNKNOWN_REQUEST;
	switch(bRequest)
	{
		case UVC_SET_CUR: bmRT = UVCC_BMRT_SET; break;
		case UVC_GET_CUR:
		case UVC_GET_MIN:
		case UVC_GET_MAX:
        case UVC_GET_RES:
        case UVC_GET_LEN:
		case UVC_GET_DEF: bmRT = UVCC_BMRT_GET; break;
		default:
			return UVCC_ERR_UNKNOWN_UVC_REQUEST;
	}
	memcpy(&req, &predef_reqs[uvccReq], sizeof(IOUSBDevRequest));
	/* TODO (maybe):
	 * check for endianness and convert PPC big endian to little
	 */
	req.bmRequestType = bmRT;
	req.bRequest = bRequest;
	req.wIndex |= cam->ifNo;
	req.pData = (void *)pData;

	if(req.wLength > 0x04) printf("what the fuck?!\n");

	return uvccSendRawRequest(cam, &req);
}

int uvccSendRawRequest(uvccCam *cam,
					   IOUSBDevRequest *request)
{
	IOReturn ior;
	//uint8_t i, nEP = 0;

	/* 0 means default pipe */
	ior = (*(cam->ctrlIf))->ControlRequest(cam->ctrlIf, 0, request);
    if(ior == kIOUSBPipeStalled)
    {
		// if(!logger) fprintf(stderr, "uvcc error! uvccSendRawRequest: Pipe indicated stall, request is probably unsupported or malformatted.");
		// else asl_log(logger, NULL, ASL_LEVEL_ERR, "uvccSendRawRequest: Pipe indicated stall, request is probably unsupported or malformatted.");
		return UVCC_ERR_CTRL_REQUEST_UNSUPPORTED_OR_MALFORMATTED;
        /*uvcc_err("uvccSendRawRequest: ControlRequest", ior);
        (*(cam->ctrlIf))->GetNumEndpoints(cam->ctrlIf, &nEP);
        for(i = 0; i <= nEP; i++)
        {
            (*(cam->ctrlIf))->AbortPipe(cam->ctrlIf, i);
            (*(cam->ctrlIf))->ClearPipeStallBothEnds(cam->ctrlIf, i);
            (*(cam->ctrlIf))->ResetPipe(cam->ctrlIf, i);
        }
        if(((*(cam->ctrlIf))->ControlRequest(cam->ctrlIf, 0, request)) == kIOUSBPipeStalled)
        {
            if(!logger) fprintf(stderr, "uvcc error! uvccSendRawRequest: Second stall directly following clear and reset, request is probably unsupported or malformatted.");
            else asl_log(logger, NULL, ASL_LEVEL_ERR, "uvccSendRawRequest: Second stall directly following clear and reset, request is probably unsupported or malformatted.");
            return UVCC_ERR_CTRL_REQUEST_UNSUPPORTED_OR_MALFORMATTED;
        } */
    }
    else if(ior == kIOReturnNotResponding)
	{	/* due to a bug in os x 10.6.8 this might be bullshit, see
		 * http://lists.apple.com/archives/usb/2011/Jun/msg00050.html
         * first lets try resetting on the device */
        (*(cam->devIf))->USBDeviceAbortPipeZero(cam->devIf);
        /* (*devIf)->USBDeviceSuspend(devIf, 0/1);
         * eller starta och stäng av..
         */
        ior = (*(cam->ctrlIf))->ControlRequest(cam->ctrlIf, 0, request);
	}
	if(ior != kIOReturnSuccess)
	{
		uvcc_err("uvccSendRawRequest: ControlRequest", ior);
		last_error = ior;
		last_error_fn = "uvccSendRawRequest";
		return UVCC_ERR_CTRL_REQUEST_FAILED;
	}
	/* maybe wLenDone should be returned insted..? */
	return 0;
}

/* this is almost a wrapper.. */
int uvccSendInfoRequest(uvccCam *cam,
						enum uvccRequest uvccReq,
						int8_t *pData)
{
    IOUSBDevRequest	req;
    if(cam->ctrlIf == NULL) return UVCC_ERR_INTERFACE_IS_NULL;
	if(uvccReq >= __UVCC_REQ_OUT_OF_RANGE) return UVCC_ERR_UNKNOWN_REQUEST;
    memcpy(&req, &predef_reqs[uvccReq], sizeof(IOUSBDevRequest));
    req.bmRequestType = UVCC_BMRT_GET;
    req.bRequest = UVC_GET_INFO;
    req.wLength = 0x01;
    req.wIndex |= cam->ifNo;
	req.pData = (void *)pData;
	return uvccSendRawRequest(cam, &req);
}

CFStringRef uvccErrorStr()
{
	CFStringRef ret;
	char *b = malloc(strlen(last_error_fn) + strlen(kio_return_str(last_error)) + 5);
	if(!b)
	{	/* error retriever causes error! */
		if(!logger) perror("uvcc error! uvccErrorStr: could not allocate memory for error string");
		else asl_log(logger, NULL, ASL_LEVEL_ERR, "uvccErrorStr: could not allocate memory for error string: %s", strerror(errno));
		return NULL;
	}
	sprintf(b, "%s(), %s", last_error_fn, kio_return_str(last_error));
	ret = CFStringCreateWithCString(kCFAllocatorDefault, b, kCFStringEncodingASCII);
	free(b);
	return ret;
}

/* this is where we put helpers.. */
int uvccMSLifeCamExposureTimeSpan(uint32_t min,
								  uint32_t max,
								  uint32_t **list)
{
	int i, lmnts = ceil(log2(max)) - ceil(log2(min));
	if(max <= min) return -1;
	(*list) = malloc(lmnts);
	if(!list)
	{	/* gosh darnit! */
		if(!logger) perror("uvcc error! uvccMSLifeCamExposureSpan: could not allocate memory for exposure time array");
		else asl_log(logger, NULL, ASL_LEVEL_ERR, "uvccMSLifeCamExposureSpan: could not allocate memory for exposure time array: %s", strerror(errno));
		return -1;
	}
	for(i = lmnts-1; i >= 0; i--)
	{	/* why they did it this way i will never understand.. */
		(*list)[i] = max;
		max /= 2;
	}
	return lmnts;
}
int uvccExposureTimeToMsLifeCamValue(uint32_t value,
									 uint32_t *msList,
									 int listLength)
{
	int i = 0, bestSoFar = 0, diff = msList[listLength - 1];
	for(; i < listLength; i++)
	{
		if(msList[i] - 1 <= value && value <= msList[i] + 1) return i;
		else if(abs(value - msList[i]) < diff) {
			diff = abs(value - msList[i]);
			bestSoFar = i;
		}
	}
	if(!logger) fprintf(stderr, "uvcc warning! uvccExposureTimeToMsLifeCamValue: Failed to find corresponding MS exposure time value for actual value %u, using closest one %u (%u)", value, bestSoFar, msList[bestSoFar]);
	else asl_log(logger, NULL, ASL_LEVEL_WARNING, "uvccExposureTimeToMsLifeCamValue: Failed to find corresponding MS exposure time value for actual value %u, using closest one %u (%u)", value, bestSoFar, msList[bestSoFar]);
	return bestSoFar;
}

/* internals! */
static CFDictionaryRef get_usb_service_dic()
{	/* this just gets the usb device matching dic */
	CFDictionaryRef mRef;
	if((mRef = IOServiceMatching(kIOUSBDeviceClassName)) == NULL)
	{   /* ASL_OPT_STDERR doesn't work on all systems.. */
		if(!logger) perror("uvcc error! get_usb_service_dic: IOServiceMatching returned NULL, no service matching dictionary for kIOUSBDeviceClassName could be created");
        else asl_log(logger, NULL, ASL_LEVEL_ERR, "get_usb_service_dic: IOServiceMatching returned NULL, no service matching dictionary for kIOUSBDeviceClassName could be created: %s", strerror(errno));
	}
	return mRef;
}

/* this is where most of the getting magic happens.. */
static int get_cam_list(io_iterator_t devIter,
						uvccCam ***list)
{
	io_iterator_t ifIter;
	kern_return_t kr;
	int i, nDevs = 0, nCams = 0;
	uvccCam **tmp;
	io_service_t devSrv, ifSrv;			/* unsigned int */
	IOCFPlugInInterface **pIf;			/* needed to find the device interface */
	HRESULT qiRes;						/* int32_t */
	int32_t score;
	IOUSBDeviceInterface197 **devIf;	/* interface to communicate with the device */

	if(!uvcc_port)
	{
		fprintf(stderr, UVCC_INIT_ERROR_MSG);
		return -1;
	}
	/* get number of devices */
	while((IOIteratorNext(devIter)) != 0) nDevs++;
	if(!nDevs) return 0;
	IOIteratorReset(devIter);
	tmp = malloc(nDevs*sizeof(uvccCam *));
	while((devSrv = IOIteratorNext(devIter)))
	{
		kr = IOCreatePlugInInterfaceForService(devSrv, kIOUSBDeviceUserClientTypeID, kIOCFPlugInInterfaceID, &pIf, &score);
		/* we're done with the devSrv */
		IOObjectRelease(devSrv);
		if(kr != kIOReturnSuccess)
		{
			uvcc_err("get_cam_list: IOCreatePlugInInterfaceForService", kr);
			IOObjectRelease(devSrv);
			continue;
		}
		qiRes = (*pIf)->QueryInterface(pIf, CFUUIDGetUUIDBytes(kIOUSBDeviceInterfaceID197), (LPVOID)&devIf);
		/* we're done with the plugin */
		(*pIf)->Stop(pIf);
		IODestroyPlugInInterface(pIf);
		/* kIOReturnSuccess is actually just 0 but since QI strictly talking
		 returns HRESULT (int32_t) and not kern_return_t (int) i figured i'll
		 just do this here */
		if(!qiRes && devIf)
		{
			kr = (*devIf)->CreateInterfaceIterator(devIf, &vid_ctrl_if_req, &ifIter);
			if(kr != kIOReturnSuccess || !IOIteratorIsValid(ifIter))
			{
				uvcc_err("get_cam_list: CreateInterfaceIterator", kr);
				IOObjectRelease(devSrv);
				continue;
			}
			if((ifSrv = IOIteratorNext(ifIter)))
			{	/* we'll just use the first vid ctrl if we get */
				if((tmp[nCams] = malloc(sizeof(uvccCam))) == NULL)
				{
					if(!logger) perror("uvcc error! get_cam_list: Could not allocate memory for list entry");
                    else asl_log(logger, NULL, ASL_LEVEL_ERR, "get_cam_list: Could not allocate memory for list entry: %s", strerror(errno));
					IOObjectRelease(ifIter);
					(*devIf)->Release(devIf);
					IOObjectRelease(devSrv);
					break;
				}
				/* set the data.. */
				if((fill_cam_struct(devIf, tmp[nCams])) != 0)
				{
					(*devIf)->Release(devIf);
					IOObjectRelease(devSrv);
					continue;
				}
				/* get the registry name */
				set_cam_reg_name(devSrv, tmp[nCams]);
				nCams++;
				IOObjectRelease(ifSrv);
			}
			/* else: no vid_ctrl_iface, i.e. not cam */
			IOObjectRelease(ifIter);
		}
		else
		{
			if(!logger) perror("uvcc warning! get_cam_list: QueryInterface failed");
            else asl_log(logger, NULL, ASL_LEVEL_WARNING, "get_cam_list: QueryInterface failed: %s", strerror(errno));
		}
		IOObjectRelease(devSrv);
	}
	if(nCams > 0)
	{	/* only make the allocation if we got cams */
		(*list) = malloc(nCams*sizeof(uvccCam *));
		for(i = 0; i < nCams; i++) (*list)[i] = tmp[i];
	}
	free(tmp);
	return nCams;
}

static int fill_cam_struct(IOUSBDeviceInterface197 **devIf,
						   uvccCam *cam)
{
    IOReturn ior;
    if((ior = (*devIf)->GetLocationID(devIf, &(cam->idLocation))) != kIOReturnSuccess)
    {
		uvcc_err("fill_cam_struct: GetLocationID", ior);
		return -1;
	}
	if((get_dev_desc(devIf, &(cam->devDesc))) != 0)
	{
		if(!logger) perror("uvcc error! fill_cam_struct: Could not retrieve device descriptor");
        else asl_log(logger, NULL, ASL_LEVEL_ERR, "fill_cam_struct: Could not retrieve device descriptor: %s", strerror(errno));
		return -1;
	}
	cam->mID = (struct uvccModelID *)&(cam->devDesc.hwIdVendor);
	cam->devIf = devIf;
	if((cam->ctrlIf = get_ctrl_if(devIf)) == NULL)
	{
		(*(cam->devIf))->Release(cam->devIf);
		CFRelease(cam->regName);
		if(!logger) perror("uvcc error! fill_cam_struct: Could not retrieve control interface");
        else asl_log(logger, NULL, ASL_LEVEL_ERR, "fill_cam_struct: Could not retrieve control interface: %s", strerror(errno));
		return -1;
	}
    if((ior = (*(cam->ctrlIf))->GetInterfaceNumber(cam->ctrlIf, &(cam->ifNo))) != kIOReturnSuccess)
    {
		(*(cam->devIf))->Release(cam->devIf);
		(*(cam->ctrlIf))->Release(cam->ctrlIf);
		uvcc_err("fill_cam_struct: GetInterfaceNumber", ior);
		return -1;
	}
	return 0;
}

static void set_cam_reg_name(io_service_t srv, uvccCam *cam)
{
	io_name_t cls;
	IOReturn ior;
	cam->regName = IORegistryEntryCreateCFProperty(srv, CFSTR(kUSBProductString), kCFAllocatorDefault, 0);
	if(cam == NULL)
	{	/* get it "manually" */
		ior = IORegistryEntryGetNameInPlane(srv, kIOServicePlane, cls);
		if(ior != kIOReturnSuccess) {
			uvcc_err("set_cam_reg_name: IORegistryEntryGetNameInPlane", ior);
			cam->regName = CFSTR(UVCC_UNKNOWN_CAM_REG_NAME);
		}
		else cam->regName = CFStringCreateWithCString(kCFAllocatorDefault, cls, kCFStringEncodingUTF8);
	}
}

static IOUSBInterfaceInterface197 **get_ctrl_if(IOUSBDeviceInterface197 **devIf)
{
	io_iterator_t				ifIter;
	IOReturn					reqRes;				/* int */
	io_service_t				camDevSrv;
	IOCFPlugInInterface			**pIf;
	int32_t						score;
	HRESULT						qiRes;
	kern_return_t				pIfKr, objRelKr;
	IOUSBInterfaceInterface197	**ctrlIf = NULL;	/* interface used to communicate with the ctrl interface */
	//int i = 0; uint8_t n;

	if(!uvcc_port)
	{
		if(!logger) fprintf(stderr, UVCC_INIT_ERROR_MSG);
        else asl_log(logger, NULL, ASL_LEVEL_ERR, UVCC_INIT_ERROR_MSG);
		return NULL;
	}
	reqRes = (*devIf)->CreateInterfaceIterator(devIf, &vid_ctrl_if_req, &ifIter);
	if(reqRes != kIOReturnSuccess)
	{
		uvcc_err("get_ctrl_if: CreateInterfaceIterator", reqRes);
		return NULL;
	}
	/* TODO:
	 * write a function that manually gets the device descriptor, loops through the configs and their settings looking for an interface with class and subclass that might work if either of theese fail (see MIDI/Shared/USBDevice.cpp, libusb/os/darwin_usb.c > process_new_device and darwin_get_config_descriptor and uvcc0.14b/uvc-controller.c -> open_uvi_index)
	 */
	//while
	if((camDevSrv = IOIteratorNext(ifIter)) != IO_OBJECT_NULL)
	{	/* once again we use an intermediate plug-in to get the interface */
		pIfKr = IOCreatePlugInInterfaceForService(camDevSrv, kIOUSBInterfaceUserClientTypeID, kIOCFPlugInInterfaceID, &pIf, &score);
		objRelKr = IOObjectRelease(camDevSrv);
		if(pIfKr != kIOReturnSuccess || !pIf)
		{
			uvcc_err("get_ctrl_if: IOCreatePlugInInterfaceForService", pIfKr);
			return NULL;
		}
		if(objRelKr != kIOReturnSuccess)
        {
            if(!logger) fprintf(stderr, "uvcc warning! get_ctrl_if: IOObjectRelease returned 0x%08x: %s.\n", objRelKr, kio_return_str(objRelKr));
            else asl_log(logger, NULL, ASL_LEVEL_WARNING, "get_ctrl_if: IOObjectRelease returned 0x%08x: %s", objRelKr, kio_return_str(objRelKr));
        }

		qiRes = (*pIf)->QueryInterface(pIf, CFUUIDGetUUIDBytes(kIOUSBInterfaceInterfaceID197), (LPVOID *)&ctrlIf);
		(*pIf)->Stop(pIf);
		IODestroyPlugInInterface(pIf);
		if(qiRes || !ctrlIf)
		{
			if(!logger) perror("uvcc error! get_ctrl_if: QueryInterface failed");
            else asl_log(logger, NULL, ASL_LEVEL_ERR, "get_ctrl_if: QueryInterface failed: %s", strerror(errno));
			return NULL;
		}
		//(*ctrlIf)->GetInterfaceNumber(ctrlIf, &n);
		//printf("ctrlIf #%i: %i\n", i++, n);
	}
	else
		//if(i == 0)
	{
		if(!logger) perror("uvcc error! get_ctrl_if: IOIteratorNext returned 0, no video control interface found on device");
        else asl_log(logger, NULL, ASL_LEVEL_ERR, "get_ctrl_if: IOIteratorNext returned 0, no video control interface found on device: %s", strerror(errno));
		return NULL;
	}
    while((camDevSrv = IOIteratorNext(ifIter)) != IO_OBJECT_NULL) printf("there's another one here!\n");
	return ctrlIf;
}

static int get_dev_desc(IOUSBDeviceInterface197 **devIf,
						struct uvccDevDesc *dd)
						//IOUSBDeviceDescriptor *dd)
{	/* i've gotten som weird ass errors using USB_REQ_GET_DESC / USB_DT_DEVICE,
	   both failed reqs that has obviously worked and successful reqs that has
	   failed to set IOUSBDeviceDescriptor data, so i created the uvccDevDesc
	   (since i don't really use all the info in the other one anyway). This
	   way there is no need to open the device to get this info */
	IOReturn ior;
	if((ior = (*devIf)->GetDeviceClass(devIf, &(dd->bDevClass))) != kIOReturnSuccess) {
		uvcc_err("get_dev_desc: GetDeviceClass", ior);
		return -1;
	}
	if((ior = (*devIf)->GetDeviceSubClass(devIf, &(dd->bDevSubClass))) != kIOReturnSuccess) {
		uvcc_err("get_dev_desc: GetDeviceSubClass", ior);
		return -1;
	}
	if((ior = (*devIf)->GetDeviceProtocol(devIf, &(dd->bDevProtocol))) != kIOReturnSuccess) {
		uvcc_err("get_dev_desc: GetDeviceProtocol", ior);
		return -1;
	}
	if((ior = (*devIf)->GetDeviceSpeed(devIf, &(dd->bDevSpeed))) != kIOReturnSuccess) {
		uvcc_err("get_dev_desc: GetDeviceSpeed", ior);
		return -1;
	}
	if((ior = (*devIf)->GetDeviceVendor(devIf, &(dd->hwIdVendor))) != kIOReturnSuccess) {
		uvcc_err("get_dev_desc: GetDeviceVendor", ior);
		return -1;
	}
	if((ior = (*devIf)->GetDeviceProduct(devIf, &(dd->hwIdProduct))) != kIOReturnSuccess) {
		uvcc_err("get_dev_desc: GetDeviceProduct", ior);
		return -1;
	}
	if((ior = (*devIf)->GetDeviceReleaseNumber(devIf, &(dd->hwRelNo))) != kIOReturnSuccess) {
		uvcc_err("get_dev_desc: GetDeviceReleaseNumber", ior);
		return -1;
	}
	if((ior = (*devIf)->GetNumberOfConfigurations(devIf, &(dd->bNumConfs))) != kIOReturnSuccess) {
		uvcc_err("get_dev_desc: GetNumberOfConfigurations", ior);
		return -1;
	}
	if((ior = (*devIf)->USBGetManufacturerStringIndex(devIf, &(dd->bIManufacturer))) != kIOReturnSuccess) {
		uvcc_err("get_dev_desc: USBGetManufacturerStringIndex", ior);
		return -1;
	}
	if((ior = (*devIf)->USBGetProductStringIndex(devIf, &(dd->bIProduct))) != kIOReturnSuccess) {
		uvcc_err("get_dev_desc: USBGetProductStringIndex", ior);
		return -1;
	}
	if((ior = (*devIf)->USBGetSerialNumberStringIndex(devIf, &(dd->bISerialNumber))) != kIOReturnSuccess) {
		uvcc_err("get_dev_desc: USBGetSerialNumberStringIndex", ior);
		return -1;
	}
	return 0;
	/* I'm still not sure if i'm gonna go back to IOUSBDeviceDescriptor, that's why i'm keeping this here.
	IOReturn ior;
	IOUSBDevRequest req;
	* start by opening device *
	ior = (*devIf)->USBDeviceOpenSeize(devIf);
	if(ior != kIOReturnSuccess) {
		uvcc_err("get_dev_desc: USBDeviceOpenSeize", ior);
		return -1;
	}

	* there are mac os x macros and consts for this.. but i don't trust em! *
	req.bmRequestType = USB_RT_TD_GET | USB_RT_TY_STD | USB_RT_RE_DEVICE;
	req.bRequest = USB_REQ_GET_DESC;
	req.wValue = (USB_DT_DEVICE << 8) | 0;
	req.wIndex = 0;
	req.pData = dd;
	req.wLength = sizeof(IOUSBDeviceDescriptor);
	ior = (*devIf)->DeviceRequest(devIf, &req);
	* iSight doesn't wanna interact if it's suspended.. also os x 10.6 usb handeling is buggy as fuck *
	if(ior == kIOReturnNotResponding)
	{
	//	uvcc_err("get_dev_desc: DeviceRequest", ior);
		* try unsuspending that bitch *
		//printf("fuck you!\n");
		(*devIf)->USBDeviceSuspend(devIf, FALSE);
		ior = (*devIf)->DeviceRequest(devIf, &req);
		(*devIf)->USBDeviceSuspend(devIf, TRUE); *
		//if(ior == kIOReturnNotResponding)
		// {	* last resort.. *
		* (*devIf)->USBDeviceOpen(devIf);
		// (*devIf)->USBDeviceReEnumerate(devIf, 0);
		 ior = (*devIf)->DeviceRequest(devIf, &req);
		 (*devIf)->USBDeviceClose(devIf); *
		 //}
		* since it's the vid_ctrl_iface we have this it's this endpoint we gotta check the status on *


	}
	(*devIf)->USBDeviceClose(devIf);
	if(ior != kIOReturnSuccess)
	{
		uvcc_err("get_dev_desc: DeviceRequest", ior);
		return -1;
	} */
}

static int get_ctrl_ep(IOUSBDeviceInterface197 **devIf)
{	/* this sucker is not used yet, but it will be when handeling
	   kIOReturnNotResponding/kIOUSBPipeStalled errors.. */
	IOReturn ior;
	uint8_t nCfgs, i, j, epStat[2];
	IOUSBDevRequest req;
	IOUSBConfigurationDescriptorPtr cfgDesc;
	/*


	 HÄMTA EP I KONSTRUKTIONEN AV camEN


	 */

	/* since a cam may be streaming we can't really use GetConfiguration */
	ior = (*devIf)->GetNumberOfConfigurations(devIf, &nCfgs);
	if(ior != kIOReturnSuccess)
	{	/* the device is prolly unplugged */
		uvcc_err("GetNumberOfConfigurations", ior);
		return -1;
	}
	/* kIOReturnNotResponding is returned willy nilly by os x so we need
	 to store the ep to see if its really no responding or just whining
	 like a little bitch... */
	for(i = 0; i < nCfgs; i++)
	{	/* going thru the descriptors (normally not that many) */
		ior = (*devIf)->GetConfigurationDescriptorPtr(devIf, i, &cfgDesc);
		if(ior != kIOReturnSuccess)
		{	/* this really shouldn't happen.. */
			uvcc_err("GetConfigurationDescriptorPtr", ior);
			continue;
		}
		for(j = 0; j < cfgDesc->bNumInterfaces; j++)
		{	/* this might be a few but the one we're looking for is more
			 often than not the first one */
			printf("desc");
		}

		/*


		 freeA cfg-descen!!!!


		 */

	}
	if(ior != kIOReturnSuccess)
	{	/* the device is prolly unplugged */
		uvcc_err("ÄNDRA DET HÄR", ior);
		return -1;
	}
	/* check endpoint status */
	req.bmRequestType = USB_RT_TD_GET | USB_RT_TY_STD | USB_RT_RE_ENDPOINT;
	req.bRequest = USB_REQ_GET_STATUS;
	req.wValue = 0;
	/* since it's the vid_ctrl_iface we have this it's this endpoint we gotta check the status on */
	(*devIf)->GetNumberOfConfigurations(devIf, &nCfgs);


	req.wIndex = 1; // the pipes endpointnumber!!!


	req.pData = &epStat;
	req.wLength = 2;

	ior = (*devIf)->DeviceRequest(devIf, &req);
	return 0;
}


static int get_string_desc(IOUSBDeviceInterface197 **devIf,
						   uint8_t index,
						   UniChar buf[128])
{
	IOUSBDevRequest	req;
	langid_arr		lid;
	IOReturn		ior;
	int				i;
	/* start by opening the device */
	ior = (*devIf)->USBDeviceOpenSeize(devIf);
	if(ior != kIOReturnSuccess) {
		uvcc_err("get_string_desc: USBDeviceOpenSeize", ior);
		return -1;
	}
	req.bmRequestType = USB_RT_TD_GET | USB_RT_TY_STD | USB_RT_RE_DEVICE;
	req.bRequest = USB_REQ_GET_DESC;
	/* first find out if we got any string descs (by getting langids) */
	req.wValue = (USB_DT_STRING << 8) | 0;
	req.wIndex = 0;
	req.pData = &lid;
	req.wLength = sizeof(lid);
	ior = (*devIf)->DeviceRequest(devIf, &req);
	if(ior != kIOReturnSuccess && ior != kIOReturnOverrun)
	{	/* apperantly overrun is normal for string descs */
		uvcc_err("get_string_desc: DeviceRequest", ior);
		return -1;
	}
	/* here we could check for a preferred lang-id arg.. we could. */
	req.wValue = (USB_DT_STRING << 8) | index;
	req.wIndex = lid.bString[0];
	/* lets just reuse lid */
	req.pData = &lid;
	req.wLength = sizeof(lid.bString);
	/* should this be done twice (second time with the given length)? */
	ior = (*devIf)->DeviceRequest(devIf, &req);
	/* close only returns error if connection is no longer valid */
	(*devIf)->USBDeviceClose(devIf);
	if(ior != kIOReturnSuccess && ior != kIOReturnOverrun)
	{	/* apperantly overrun is normal for string descs */
		uvcc_err("get_string_desc: DeviceRequest", ior);
		return -1;
	}
	for(i = 0; i < (lid.bLength-2)/2; i++) buf[i] = USBToHostWord(lid.bString[i]);
	/* returned strings are not null terminated */
	buf[i] = 0;
	return (lid.bLength-2)/2;
}

static void uvcc_err(const char *f,
					 int r)
{
	if(!logger) fprintf(stderr, "uvcc error! %s returned 0x%08x %s.\n", f, r, kio_return_str(r));
    else asl_log(logger, NULL, ASL_LEVEL_ERR, "%s returned 0x%08x %s.\n", f, r, kio_return_str(r));
}

/* i was lazy and just parsed IOKit/IOReturn.h and IOKit/usb/USB.h
 * TODO: fix duplicate case values
 */
static const char *kio_return_str(int r)
{
	switch(r)
	{
		case kIOReturnSuccess:
			return "(kIOReturnSuccess): OK";
		case kIOReturnError:
			return "(kIOReturnError): general error";
		case kIOReturnNoMemory:
			return "(kIOReturnNoMemory): can't allocate memory";
		case kIOReturnNoResources:
			return "(kIOReturnNoResources): resource shortage";
		case kIOReturnIPCError:
			return "(kIOReturnIPCError): error during IPC";
		case kIOReturnNoDevice:
			return "(kIOReturnNoDevice): no such device";
		case kIOReturnNotPrivileged:
			return "(kIOReturnNotPrivileged): privilege violation";
		case kIOReturnBadArgument:
			return "(kIOReturnBadArgument): invalid argument";
		case kIOReturnLockedRead:
			return "(kIOReturnLockedRead): device read locked";
		case kIOReturnLockedWrite:
			return "(kIOReturnLockedWrite): device write locked";
		case kIOReturnExclusiveAccess:
			return "(kIOReturnExclusiveAccess): exclusive access and device already open";
		case kIOReturnBadMessageID:
			return "(kIOReturnBadMessageID): sent/received messages had different msg_id";
		case kIOReturnUnsupported:
			return "(kIOReturnUnsupported): unsupported function";
		case kIOReturnVMError:
			return "(kIOReturnVMError): misc. VM failure";
		case kIOReturnInternalError:
			return "(kIOReturnInternalError): internal error";
		case kIOReturnIOError:
			return "(kIOReturnIOError): General I/O error";
		case kIOReturnCannotLock:
			return "(kIOReturnCannotLock): can't acquire lock";
		case kIOReturnNotOpen:
			return "(kIOReturnNotOpen): device not open";
		case kIOReturnNotReadable:
			return "(kIOReturnNotReadable): read not supported";
		case kIOReturnNotWritable:
			return "(kIOReturnNotWritable): write not supported";
		case kIOReturnNotAligned:
			return "(kIOReturnNotAligned): alignment error";
		case kIOReturnBadMedia:
			return "(kIOReturnBadMedia): Media Error";
		case kIOReturnStillOpen:
			return "(kIOReturnStillOpen): device(s) still open";
		case kIOReturnRLDError:
			return "(kIOReturnRLDError): rld failure";
		case kIOReturnDMAError:
			return "(kIOReturnDMAError): DMA failure";
		case kIOReturnBusy:
			return "(kIOReturnBusy): Device Busy";
		case kIOReturnTimeout:
			return "(kIOReturnTimeout): I/O Timeout";
		case kIOReturnOffline:
			return "(kIOReturnOffline): device offline";
		case kIOReturnNotReady:
			return "(kIOReturnNotReady): not ready";
		case kIOReturnNotAttached:
			return "(kIOReturnNotAttached): device not attached";
		case kIOReturnNoChannels:
			return "(kIOReturnNoChannels): no DMA channels left";
		case kIOReturnNoSpace:
			return "(kIOReturnNoSpace): no space for data";
		case kIOReturnPortExists:
			return "(kIOReturnPortExists): port already exists";
		case kIOReturnCannotWire:
			return "(kIOReturnCannotWire): can't wire down physical memory";
		case kIOReturnNoInterrupt:
			return "(kIOReturnNoInterrupt): no interrupt attached";
		case kIOReturnNoFrames:
			return "(kIOReturnNoFrames): no DMA frames enqueued";
		case kIOReturnMessageTooLarge:
			return "(kIOReturnMessageTooLarge): oversized msg received on interrupt port";
		case kIOReturnNotPermitted:
			return "(kIOReturnNotPermitted): not permitted";
		case kIOReturnNoPower:
			return "(kIOReturnNoPower): no power to device";
		case kIOReturnNoMedia:
			return "(kIOReturnNoMedia): media not present";
		case kIOReturnUnformattedMedia:
			return "(kIOReturnUnformattedMedia): media not formatted";
		case kIOReturnUnsupportedMode:
			return "(kIOReturnUnsupportedMode): no such mode";
		case kIOReturnUnderrun:
			return "(kIOReturnUnderrun): data underrun";
		case kIOReturnOverrun:
			return "(kIOReturnOverrun): data overrun";
		case kIOReturnDeviceError:
			return "(kIOReturnDeviceError): the device is not working properly!";
		case kIOReturnNoCompletion:
			return "(kIOReturnNoCompletion): a completion routine is required";
		case kIOReturnAborted:
			return "(kIOReturnAborted): operation aborted";
		case kIOReturnNoBandwidth:
			return "(kIOReturnNoBandwidth): bus bandwidth would be exceeded";
		case kIOReturnNotResponding:
			return "(kIOReturnNotResponding): device not responding";
		case kIOReturnIsoTooOld:
			return "(kIOReturnIsoTooOld): isochronous I/O request for distant past!";
		case kIOReturnIsoTooNew:
			return "(kIOReturnIsoTooNew): isochronous I/O request for distant future";
		case kIOReturnNotFound:
			return "(kIOReturnNotFound): data was not found";
		case kIOReturnInvalid:
			return "(kIOReturnInvalid): you just broke the internet..";
		case kIOUSBUnknownPipeErr:
			return "(kIOUSBUnknownPipeErr): Pipe ref not recognized";
		case kIOUSBTooManyPipesErr:
			return "(kIOUSBTooManyPipesErr): Too many pipes";
		case kIOUSBNoAsyncPortErr:
			return "(kIOUSBNoAsyncPortErr): no async port";
		case kIOUSBNotEnoughPipesErr:
			return "(kIOUSBNotEnoughPipesErr): not enough pipes in interface";
		case kIOUSBNotEnoughPowerErr:
			return "(kIOUSBNotEnoughPowerErr): not enough power for selected configuration";
		case kIOUSBEndpointNotFound:
			return "(kIOUSBEndpointNotFound): Endpoint Not found";
		case kIOUSBConfigNotFound:
			return "(kIOUSBConfigNotFound): Configuration Not found";
		case kIOUSBTransactionTimeout:
			return "(kIOUSBTransactionTimeout): Transaction timed out";
		case kIOUSBTransactionReturned:
			return "(kIOUSBTransactionReturned): The transaction has been returned to the caller";
		case kIOUSBPipeStalled:
			return "(kIOUSBPipeStalled): Pipe has stalled, error needs to be cleared";
		case kIOUSBInterfaceNotFound:
			return "(kIOUSBInterfaceNotFound): Interface ref not recognized";
		case kIOUSBLowLatencyBufferNotPreviouslyAllocated:
			return "(kIOUSBLowLatencyBufferNotPreviouslyAllocated): Attempted to use user land low latency isoc calls w/out calling PrepareBuffer (on the data buffer) first";
		case kIOUSBLowLatencyFrameListNotPreviouslyAllocated:
			return "(kIOUSBLowLatencyFrameListNotPreviouslyAllocated): Attempted to use user land low latency isoc calls w/out calling PrepareBuffer (on the frame list) first";
		case kIOUSBHighSpeedSplitError:
			return "(kIOUSBHighSpeedSplitError): Error to hub on high speed bus trying to do split transaction";
		case kIOUSBSyncRequestOnWLThread:
			return "(kIOUSBSyncRequestOnWLThread): A synchronous USB request was made on the workloop thread (from a callback?).  Only async requests are permitted in that case";
		case kIOUSBDeviceNotHighSpeed:
			return "(kIOUSBDeviceNotHighSpeed): The device is not a high speed device, so the EHCI driver returns an error";
/*		case kIOUSBDevicePortWasNotSuspended:
			return "(kIOUSBDevicePortWasNotSuspended): Port was not suspended"; */
		case kIOUSBLinkErr:
			return "(kIOUSBLinkErr): Link error";
		case kIOUSBNotSent2Err:
			return "(kIOUSBNotSent2Err): Transaction not sent";
		case kIOUSBNotSent1Err:
			return "(kIOUSBNotSent1Err): Transaction not sent";
		case kIOUSBBufferUnderrunErr:
			return "(kIOUSBBufferUnderrunErr): Buffer Underrun (Host hardware failure on data out, PCI busy?)";
		case kIOUSBBufferOverrunErr:
			return "(kIOUSBBufferOverrunErr): Buffer Overrun (Host hardware failure on data out, PCI busy?)";
		case kIOUSBReserved2Err:
			return "(kIOUSBReserved2Err): Reserved";
		case kIOUSBReserved1Err:
			return "(kIOUSBReserved1Err): Reserved";
		case kIOUSBWrongPIDErr:
			return "(kIOUSBWrongPIDErr): Pipe stall, Bad or wrong PID";
		case kIOUSBPIDCheckErr:
			return "(kIOUSBPIDCheckErr): Pipe stall, PID CRC error";
		case kIOUSBDataToggleErr:
			return "(kIOUSBDataToggleErr): Pipe stall, Bad data toggle";
		case kIOUSBBitstufErr:
			return "(kIOUSBBitstufErr): Pipe stall, bitstuffing";
		case kIOUSBCRCErr:
			return "(kIOUSBCRCErr): Pipe stall, bad CRC";
/*		case kIOUSBMessageHubResetPort:
			return "(kIOUSBMessageHubResetPort): Message sent to a hub to reset a particular port";
        case kIOUSBMessageHubSuspendPort:
			return "(kIOUSBMessageHubSuspendPort): Message sent to a hub to suspend a particular port";
        case kIOUSBMessageHubResumePort:
			return "(kIOUSBMessageHubResumePort): Message sent to a hub to resume a particular port"; */
		case kIOUSBMessageHubIsDeviceConnected:
			return "(kIOUSBMessageHubIsDeviceConnected): Message sent to a hub to inquire whether a particular port has a device connected or not";
		case kIOUSBMessageHubIsPortEnabled:
			return "(kIOUSBMessageHubIsPortEnabled): Message sent to a hub to inquire whether a particular port is enabled or not";
/*		case kIOUSBMessageHubReEnumeratePort:
			return "(kIOUSBMessageHubReEnumeratePort): Message sent to a hub to reenumerate the device attached to a particular port";
        case kIOUSBMessagePortHasBeenReset:
			return "(kIOUSBMessagePortHasBeenReset): Message sent to a device indicating that the port it is attached to has been reset";
        case kIOUSBMessagePortHasBeenResumed:
			return "(kIOUSBMessagePortHasBeenResumed): Message sent to a device indicating that the port it is attached to has been resumed";
        case kIOUSBMessageHubPortClearTT:
			return "(kIOUSBMessageHubPortClearTT): Message sent to a hub to clear the transaction translator";
        case kIOUSBMessagePortHasBeenSuspended:
			return "(kIOUSBMessagePortHasBeenSuspended): Message sent to a device indicating that the port it is attached to has been suspended";
        case kIOUSBMessageFromThirdParty:
			return "(kIOUSBMessageFromThirdParty): Message sent from a third party.  Uses IOUSBThirdPartyParam to encode the sender's ID";
        case kIOUSBMessagePortWasNotSuspended:
			return "(kIOUSBMessagePortWasNotSuspended): Message indicating that the hub driver received a resume request for a port that was not suspended";
        case kIOUSBMessageExpressCardCantWake:
			return "(kIOUSBMessageExpressCardCantWake): Message from a driver to a bus that an express card will disconnect on sleep and thus shouldn't wake"; */
		case kIOUSBMessageCompositeDriverReconfigured:
			return "(kIOUSBMessageCompositeDriverReconfigured): Message from the composite driver indicating that it has finished re-configuring the device after a reset";
		case kIOUSBMessageHubSetPortRecoveryTime:
			return "(kIOUSBMessageHubSetPortRecoveryTime): Message sent to a hub to set the # of ms required when resuming a particular port";
		case kIOUSBMessageOvercurrentCondition:
			return "(kIOUSBMessageOvercurrentCondition): Message sent to the clients of the device's hub parent, when a device causes an overcurrent condition.  The message argument contains the locationID of the device";
		case kIOUSBMessageNotEnoughPower:
			return "(kIOUSBMessageNotEnoughPower): Message sent to the clients of the device's hub parent, when a device causes an low power notice to be displayed.  The message argument contains the locationID of the device";
		case kIOUSBMessageController:
			return "(kIOUSBMessageController): Generic message sent from controller user client to controllers";
		case kIOUSBMessageRootHubWakeEvent:
			return "(kIOUSBMessageRootHubWakeEvent): Message from the HC Wakeup code indicating that a Root Hub port has a wake event";
		default:
			return "(kIOYouSuck): Unknown error.. that really shouldn't happen";
	}
}
