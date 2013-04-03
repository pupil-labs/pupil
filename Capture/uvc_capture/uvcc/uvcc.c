/* by sluggo, sluggo@dm9.se ( http://dm9.se )
 * NOTE:
 * This lib in no way implements all of the uvc-standard. It's main purpose is help people control some of the basic functions available in their webcams.
 *
 * TODO:
 *  - read up on PU's.. why doesn't ms-cams work?
 *  - get dims @ fps
 *	- more error-codes
 *  - fix handling of not responding devices
 *  - [see source]
 *
 * CODE NOTES:
 * using IOUSBDeviceInterface197 and IOUSBInterfaceInterface197 which means IOUSBFamily 1.9.7 or above is required, that is os x >= 10.2.5. if anyone wants it's most certainly possible to rewrite for earlier versions.
 * the code looks a bit incoherent, think of it as everything that is directly callable by a user is ObjC-formatted and the other stuff is more c. just wanted to point out that im not a big fan of oop. not a big fan at all..
 * look at "Mac OS X 10.6 Core Library > Drivers, Kernel & Hardware > User-Space Device Access > USB Device Inteface Guide" (especially the SampleUSBMIDIDriver and Deva_Example projects), Dominic Szablewski's uvc camera control ( http://www.phoboslab.org/log/2009/07/uvc-camera-control-for-mac-os-x ) and the libusb project ( http://libusb.org ) to see where this code came from..
 */



// moritz kassner: uncommented 405/415/416
#include <IOKit/IOKitLib.h>
#include <IOKit/IOCFPlugIn.h>
#include <mach/mach_init.h>
#include <mach/mach_port.h>
#include <asl.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "uvcc.h"

#define UVCC_INIT_ERROR_MSG "You have to call uvccInit first dummy!\n"

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
    /* TODO: UVC_CT_ZOOM_ABSOLUTE_CONTROL */
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
    /* TODO: UVC_PU_HUE_AUTO_CONTROL */
    /* TODO: UVC_PU_HUE_CONTROL (also fucking signed) */
	{ 0,0, UVC_PU_SATURATION_CONTROL, UVC_PU_INDEX, 0x02, NULL, 0 },
	{ 0,0, UVC_PU_SHARPNESS_CONTROL, UVC_PU_INDEX, 0x02, NULL, 0 },
    { 0,0, UVC_PU_GAMMA_CONTROL, UVC_PU_INDEX, 0x02, NULL, 0 },
	{ 0,0, UVC_PU_WHITE_BALANCE_TEMPERATURE_AUTO_CONTROL, UVC_PU_INDEX, 0x01, NULL, 0 },
	{ 0,0, UVC_PU_WHITE_BALANCE_TEMPERATURE_CONTROL, UVC_PU_INDEX, 0x02, NULL, 0 }
    /* TODO: UVC_PU_WHITE_BALANCE_COMPONENT_AUTO_CONTROL */
    /* TODO: UVC_PU_WHITE_BALANCE_COMPONENT_CONTROL */
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
	UInt8	bLength;
	UInt8	bDescriptorType;
	UniChar	bString[127];	/* UInt16 */
} str_desc;
typedef str_desc langid_arr;

/* we'll need our own iokit port otherwise everythings goes to shit when trying to communicate with more than one cam */
static mach_port_t	uvcc_port = 0;
/* the logger */
static aslclient logger;

/* here are some internal funcs, listed at the bottom */
static int fill_cam_struct(IOUSBDeviceInterface197 **devIf,
						   struct uvccCam *cam);
static IOUSBInterfaceInterface197 **get_ctrl_if(IOUSBDeviceInterface197 **devIf);
static int get_dev_desc(IOUSBDeviceInterface197 **devIf,
						IOUSBDeviceDescriptor *dd);
static int get_string_desc(IOUSBDeviceInterface197 **devIf,
						   UInt8 index,
						   UniChar buf[128]);
static void uvcc_err(const char *f,
					 int r);
static const char *kio_return_str(int r);

/* funcs are listed in the same order as in the header */
int uvccInit()
{
    int ret = UVCCE_NO_ERROR;
    /* mach_port_null -> default port to communicate with iokit with */
	kern_return_t kr = IOMasterPort(MACH_PORT_NULL, &uvcc_port);
    logger = asl_open("se.dm9.uvcc", "uvcc logger facility", ASL_OPT_STDERR | ASL_OPT_NO_DELAY);
    if(!logger) ret = UVCCW_LOGGING_TO_STDERR;
	if(kr != kIOReturnSuccess || !uvcc_port)
	{
		uvcc_err("IOMasterPort", kr);
		return UVCCE_CREATE_MASTER_PORT_FAIL;
	}
	return ret;
}

void uvccExit()
{
	if(uvcc_port) mach_port_deallocate(mach_task_self(), uvcc_port);
    if(logger) asl_close(logger);
}

int uvccGetCamList(struct uvccCam ***list)
{
	io_iterator_t			devIter, ifIter;	/* unsigned int */
	kern_return_t			kr;					/* int */
	int						i, nDevs = 0, nCams = 0;
	struct uvccCam			**tmp;
	io_service_t			devSrv, ifSrv;		/* unsigned int */
	IOCFPlugInInterface		**pIf;				/* needed to find the device interface */
	HRESULT					qiRes;				/* SInt32 */
	SInt32					score;
	IOUSBDeviceInterface197	**devIf;			/* interface to communicate with the device */

	if(!uvcc_port)
	{
		fprintf(stderr, UVCC_INIT_ERROR_MSG);
		return -1;
	}
	kr = IOServiceGetMatchingServices(uvcc_port, IOServiceMatching(kIOUSBDeviceClassName), &devIter);
	if(kr != kIOReturnSuccess || !IOIteratorIsValid(devIter))
	{
		uvcc_err("IOServiceGetMatchingServices", kr);
		return -1;
	}
	while((IOIteratorNext(devIter)) != 0) nDevs++;
	if(!nDevs) return 0;
	IOIteratorReset(devIter);
	tmp = malloc(nDevs*sizeof(struct uvccCam *));
	while((devSrv = IOIteratorNext(devIter)))
	{
		kr = IOCreatePlugInInterfaceForService(devSrv, kIOUSBDeviceUserClientTypeID, kIOCFPlugInInterfaceID, &pIf, &score);
		/* we're done with the devSrv */
		IOObjectRelease(devSrv);
		if(kr != kIOReturnSuccess)
		{
			uvcc_err("IOCreatePlugInInterfaceForService", kr);
			continue;
		}
		qiRes = (*pIf)->QueryInterface(pIf, CFUUIDGetUUIDBytes(kIOUSBDeviceInterfaceID197), (LPVOID)&devIf);
		/* we're done with the plugin */
		(*pIf)->Stop(pIf);
		IODestroyPlugInInterface(pIf);
		/* kIOReturnSuccess is actually just 0 but since QI strictly talking returns HRESULT (SInt32) and not kern_return_t (int) i figured i'll just do this here */
		if(!qiRes && devIf)
		{
			kr = (*devIf)->CreateInterfaceIterator(devIf, &vid_ctrl_if_req, &ifIter);
			if(kr != kIOReturnSuccess || !IOIteratorIsValid(ifIter))
			{
				uvcc_err("CreateInterfaceIterator", kr);
				continue;
			}
			if((ifSrv = IOIteratorNext(ifIter)))
			{
				if((tmp[nCams] = malloc(sizeof(struct uvccCam))) == NULL)
				{
					if(!logger) perror("uvcc error: Could not allocate space for list entry");
                    else asl_log(logger, NULL, ASL_LEVEL_ERR, "Could not allocate space for list entry: %s", strerror(errno));
					IOObjectRelease(ifIter);
					(*devIf)->Release(devIf);
					break;
				}
				if((fill_cam_struct(devIf, tmp[nCams])) != 0)
				{
					(*devIf)->Release(devIf);
					continue;
				}
				nCams++;
			}
			IOObjectRelease(ifIter);
			//(*devIf)->Release(devIf);
		}
		else
		{
			if(!logger) perror("uvcc warning: QueryInterface failed");
            else asl_log(logger, NULL, ASL_LEVEL_WARNING, "QueryInterface failed: %s", strerror(errno));
			continue;
		}
	}
	IOObjectRelease(devIter);
	(*list) = malloc(nCams*sizeof(struct uvccCam *));
	for(i = 0; i < nCams; i++) (*list)[i] = tmp[i];
	free(tmp);
	return nCams;
}

void uvccReleaseCamList(struct uvccCam **list,
					 int len)
{
	int i;
	for(i = 0; i < len; i++)
	{
		(*(list[i]->devIf))->Release(list[i]->devIf);
        (*(list[i]->ctrlIf))->Release(list[i]->ctrlIf);
		free(list[i]);
	}
	free(list);
}

int uvccGetCamWithQTUniqueID(char *uID, struct uvccCam *cam)
{
	struct uvccModelId mId;
	char idVendor[5], idProduct[5];
	if(strlen(uID) != 18 || uID[0] != '0' || uID[1] != 'x')
	{
		if(!logger) fprintf(stderr, "uvcc error: supplied string is not an QTKit unique ID\n");
        else asl_log(logger, NULL, ASL_LEVEL_ERR, "supplied string is not an QTKit unique ID");
		return -1;
	}
	strncpy(idVendor, &uID[10], 4);
	strncpy(idProduct, &uID[14], 4);
	mId.idVendor = atoi(idVendor);
	mId.idProduct = atoi(idProduct);
	return uvccGetCamWithModelId(&mId, cam);
}

int uvccGetCamWithModelId(struct uvccModelId *mId, struct uvccCam *cam)
{
	CFNumberRef				nRef;
	io_service_t			camSrv;
	IOUSBDeviceInterface197	**devIf;
	IOCFPlugInInterface		**pIf;
	kern_return_t			pIfKr;
	SInt32					score;
	HRESULT					qiRes;
	CFMutableDictionaryRef	mdRef;

	if(!uvcc_port)
	{
		fprintf(stderr, UVCC_INIT_ERROR_MSG);
		return -1;
	}
	/* set up the matching criteria for the device service we want */
	if((mdRef = IOServiceMatching(kIOUSBDeviceClassName)) == NULL)
	{   /* ASL_OPT_STDERR doesn't work on all systems.. */
		if(!logger) perror("uvcc error: IOServiceMatching returned NULL, no service matching kIOUSBDeviceClassName was found");
        else asl_log(logger, NULL, ASL_LEVEL_ERR, "IOServiceMatching returned NULL, no service matching kIOUSBDeviceClassName was found: %s", strerror(errno));
		return -1;
		// should handle error
	}
	nRef = CFNumberCreate(kCFAllocatorDefault, kCFNumberLongType, &(mId->idVendor));
	CFDictionarySetValue(mdRef, CFSTR(kUSBVendorID), nRef);
	CFRelease(nRef);
	nRef = CFNumberCreate(kCFAllocatorDefault, kCFNumberLongType, &(mId->idProduct));
	CFDictionarySetValue(mdRef, CFSTR(kUSBProductID), nRef);
	CFRelease(nRef);
	camSrv = IOServiceGetMatchingService(uvcc_port, mdRef);
	pIfKr = IOCreatePlugInInterfaceForService(camSrv, kIOUSBDeviceUserClientTypeID, kIOCFPlugInInterfaceID, &pIf, &score);
	if(pIfKr != kIOReturnSuccess || !pIf)
	{
		uvcc_err("IOCreatePlugInInterfaceForService", pIfKr);
		return -1;
	}
	qiRes = (*pIf)->QueryInterface(pIf, CFUUIDGetUUIDBytes(kIOUSBDeviceInterfaceID), (LPVOID *)&devIf);
	(*pIf)->Release(pIf);
	if(qiRes || !devIf)
	{
		if(!logger) perror("uvcc error: QueryInterface failed");
        else asl_log(logger, NULL, ASL_LEVEL_ERR, "QueryInterface failed: %s", strerror(errno));
		return -1;
	}
	/* clean up */
	(*pIf)->Stop(pIf);
	IODestroyPlugInInterface(pIf);
	/* fill and return */
	if((fill_cam_struct(devIf, cam)) != 0)
	{
		(*devIf)->Release(devIf);
		return -1;
	}
	return 0;
}

void uvccReleaseCam(struct uvccCam *cam)
{
	(*(cam->devIf))->Release(cam->devIf);
    (*(cam->ctrlIf))->Release(cam->ctrlIf);
}

char *uvccCamQTUniqueID(struct uvccCam *cam, char buf[19])
{
	sprintf(buf, "0x%08lux%04x%04x", cam->idLocation, cam->mId->idVendor, cam->mId->idProduct);
	return buf;
}

int uvccUni2Char(UniChar *src, char *dst, int len, int errChar)
{
	int si, di;
	UInt8 *_src = (UInt8 *)src;
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

int uvccCamManufacturer(struct uvccCam *cam, UniChar buf[128])
{
	return get_string_desc(cam->devIf, cam->devDesc.iManufacturer, buf);
}
int uvccCamProduct(struct uvccCam *cam, UniChar buf[128])
{
	return get_string_desc(cam->devIf, cam->devDesc.iProduct, buf);
}
int uvccCamSerialNumber(struct uvccCam *cam, UniChar buf[128])
{
	return get_string_desc(cam->devIf, cam->devDesc.iSerialNumber, buf);
}

/*



 Lägg in resolution request!!



 */

unsigned int uvccSendRequest(struct uvccCam *cam,
							 UInt8 bRequest,
							 unsigned int uvccRequest,
							 void *pData)
{
	IOUSBDevRequest	req;
	UInt8			bmRT;
	if(cam == NULL) return UVCCE_CAM_IS_NULL;
	else if(cam->ctrlIf == NULL) return UVCCE_INTERFACE_IS_NULL;
	if(uvccRequest >= __UVCC_REQ_OUT_OF_RANGE) return UVCCE_UNKNOWN_REQUEST;
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
			return UVCCE_UNKNOWN_UVC_REQUEST;
	}
	memcpy(&req, &predef_reqs[uvccRequest], sizeof(IOUSBDevRequest));
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

unsigned int uvccSendRawRequest(struct uvccCam *cam,
								IOUSBDevRequest *request)
{
	IOReturn ior;
	UInt8 i, nEP = 0;

	/* 0 means default pipe */
	ior = (*(cam->ctrlIf))->ControlRequest(cam->ctrlIf, 0, request);
	//ior = (*ctrlIf)->ControlRequest(ctrlIf, 0, request);
    if(ior == kIOUSBPipeStalled)
    {
        // uvcc_err("ControlRequest", ior);
        (*(cam->ctrlIf))->GetNumEndpoints(cam->ctrlIf, &nEP);
        for(i = 0; i <= nEP; i++)
        {
            (*(cam->ctrlIf))->AbortPipe(cam->ctrlIf, i);
            (*(cam->ctrlIf))->ClearPipeStallBothEnds(cam->ctrlIf, i);
            (*(cam->ctrlIf))->ResetPipe(cam->ctrlIf, i);
        }
        if(((*(cam->ctrlIf))->ControlRequest(cam->ctrlIf, 0, request)) == kIOUSBPipeStalled)
        {
            // if(!logger) fprintf(stderr, "uvcc error: Second stall directly following clear/reset, request is probably unsupported or malformatted.");
            // else asl_log(logger, NULL, ASL_LEVEL_ERR, "Second stall directly following clear/reset, request is probably unsupported or malformatted.");
            return UVCCE_CTRL_REQUEST_UNSUPPORTED_OR_MALFORMATTED;
        }
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
		uvcc_err("ControlRequest", ior);
		return UVCCE_CTRL_REQUEST_FAILED;
	}
	/* maybe wLenDone should be returned insted..? */
	return 0;
}

/* this is almost a wrapper.. */
int8_t uvccRequestInfo(struct uvccCam *cam,
                       unsigned int uvccRequest)
{
    int8_t val;
    IOUSBDevRequest	req;
    int ret;
    if(cam->ctrlIf == NULL) return UVCCE_INTERFACE_IS_NULL;
	if(uvccRequest >= __UVCC_REQ_OUT_OF_RANGE) return UVCCE_UNKNOWN_REQUEST;
    memcpy(&req, &predef_reqs[uvccRequest], sizeof(IOUSBDevRequest));
    req.bmRequestType = UVCC_BMRT_GET;
    req.bRequest = UVC_GET_INFO;
    req.wLength = 0x01;
    req.wIndex |= cam->ifNo;
	req.pData = (void *)&val;
    if((ret = uvccSendRawRequest(cam, &req)) != 0) return ret;
    else return val;
}

/* internals! */
static int fill_cam_struct(IOUSBDeviceInterface197 **devIf,
						   struct uvccCam *cam)
{
    IOReturn ior;
    if((ior = (*devIf)->GetLocationID(devIf, &(cam->idLocation))) != kIOReturnSuccess)
    {
		uvcc_err("GetLocationID", ior);
		return -1;
	}
	if((get_dev_desc(devIf, &(cam->devDesc))) != 0)
	{
		if(!logger) perror("uvcc error: Could not retrieve device descriptor");
        else asl_log(logger, NULL, ASL_LEVEL_ERR, "Could not retrieve device descriptor: %s", strerror(errno));
		return -1;
	}
	cam->mId = (struct uvccModelId *)&(cam->devDesc.idVendor);
	cam->devIf = devIf;
	if((cam->ctrlIf = get_ctrl_if(devIf)) == NULL)
	{
		if(!logger) perror("uvcc error: Could not retrieve control interface");
        else asl_log(logger, NULL, ASL_LEVEL_ERR, "Could not retrieve control interface: %s", strerror(errno));
		return -1;
	}
    if((ior = (*(cam->ctrlIf))->GetInterfaceNumber(cam->ctrlIf, &(cam->ifNo))) != kIOReturnSuccess)
    {
		uvcc_err("GetInterfaceNumber", ior);
		return -1;
	}
	return 0;
}

static IOUSBInterfaceInterface197 **get_ctrl_if(IOUSBDeviceInterface197 **devIf)
{
	io_iterator_t				ifIter;
	IOReturn					reqRes;				/* int */
	io_service_t				camDevSrv;
	IOCFPlugInInterface			**pIf;
	SInt32						score;
	HRESULT						qiRes;
	kern_return_t				pIfKr, objRelKr;
	IOUSBInterfaceInterface197	**ctrlIf = NULL;	/* interface used to communicate with the ctrl interface on the device */
	//int i = 0; UInt8 n;

	if(!uvcc_port)
	{
		if(!logger) fprintf(stderr, UVCC_INIT_ERROR_MSG);
        else asl_log(logger, NULL, ASL_LEVEL_ERR, UVCC_INIT_ERROR_MSG);
		return NULL;
	}
	reqRes = (*devIf)->CreateInterfaceIterator(devIf, &vid_ctrl_if_req, &ifIter);
	if(reqRes != kIOReturnSuccess)
	{
		uvcc_err("CreateInterfaceIterator", reqRes);
		return NULL;
	}
	/* TODO:
	 * write function that manually gets the device descriptor, loops through the configs and their settings looking for an interface with class and subclass that might work if either of theese fail (see MIDI/Shared/USBDevice.cpp, libusb/os/darwin_usb.c > process_new_device and darwin_get_config_descriptor and uvcc0.14b/uvc-controller.c -> open_uvi_index)
	 */
	//while
	if((camDevSrv = IOIteratorNext(ifIter)) != IO_OBJECT_NULL)
	{
		/* once again we use an intermediate plug-in to get the interface */
		pIfKr = IOCreatePlugInInterfaceForService(camDevSrv, kIOUSBInterfaceUserClientTypeID, kIOCFPlugInInterfaceID, &pIf, &score);
		objRelKr = IOObjectRelease(camDevSrv);
		if(pIfKr != kIOReturnSuccess || !pIf)
		{
			uvcc_err("IOCreatePlugInInterfaceForService", pIfKr);
			return NULL;
		}
		if(objRelKr != kIOReturnSuccess)
        {
            if(!logger) fprintf(stderr, "uvcc warning: IOObjectRelease returned 0x%08x: %s.\n", objRelKr, kio_return_str(objRelKr));
            else asl_log(logger, NULL, ASL_LEVEL_WARNING, "IOObjectRelease returned 0x%08x: %s", objRelKr, kio_return_str(objRelKr));
        }

		qiRes = (*pIf)->QueryInterface(pIf, CFUUIDGetUUIDBytes(kIOUSBInterfaceInterfaceID197), (LPVOID *)&ctrlIf);
		IODestroyPlugInInterface(pIf);
		if(qiRes || !ctrlIf)
		{
			if(!logger) perror("uvcc error: QueryInterface failed");
            else asl_log(logger, NULL, ASL_LEVEL_ERR, "QueryInterface failed: %s", strerror(errno));
			return NULL;
		}
		//(*ctrlIf)->GetInterfaceNumber(ctrlIf, &n);
		//printf("ctrlIf #%i: %i\n", i++, n);
	}
	else
		//if(i == 0)
	{
		if(!logger) perror("uvcc error: IOIteratorNext returned 0, no video control interface found on device");
        else asl_log(logger, NULL, ASL_LEVEL_ERR, "IOIteratorNext returned 0, no video control interface found on device: %s", strerror(errno));
		return NULL;
	}
    while((camDevSrv = IOIteratorNext(ifIter)) != IO_OBJECT_NULL) printf("there's another one here!\n");
	return ctrlIf;
}

static int get_dev_desc(IOUSBDeviceInterface197 **devIf,
						IOUSBDeviceDescriptor *dd)
{
	IOReturn ior;
	IOUSBDevRequest req;
	/* there are mac os x macros and consts for this.. but i don't trust em! */
	req.bmRequestType = USB_RT_TD_GET | USB_RT_TY_STD | USB_RT_RE_DEVICE;
	req.bRequest = USB_REQ_GET_DESC;
	req.wValue = (USB_DT_DEVICE << 8) | 0;
	req.wIndex = 0;
	req.pData = dd;
	req.wLength = sizeof(IOUSBDeviceDescriptor);
	ior = (*devIf)->DeviceRequest(devIf, &req);
	/* iSight doesn't wanna interact if it's suspended.. also os x 10.6 usb handeling is buggy as fuck */
	if(ior == kIOReturnNotResponding)
	{	 /* try unsuspending that bitch */
		(*devIf)->USBDeviceSuspend(devIf, FALSE);
		ior = (*devIf)->DeviceRequest(devIf, &req);
		(*devIf)->USBDeviceSuspend(devIf, TRUE);
		// if(ior == kIOReturnNotResponding)
		// {	/* last resort.. */
  //           printf("Not awake \n");
		// (*devIf)->USBDeviceOpen(devIf);
		// (*devIf)->USBDeviceReEnumerate(devIf, 0);
		//  ior = (*devIf)->DeviceRequest(devIf, &req);
		//  (*devIf)->USBDeviceClose(devIf);
		//  }
	}
	if(ior != kIOReturnSuccess)
	{
		uvcc_err("DeviceRequest", ior);
		return -1;
	}
	return 0;
}

static int get_string_desc(IOUSBDeviceInterface197 **devIf,
						   UInt8 index,
						   UniChar buf[128])
{
	IOUSBDevRequest	req;
	langid_arr		lid;
	IOReturn		ior;
	int				i;
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
		uvcc_err("DeviceRequest", ior);
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
	if(ior != kIOReturnSuccess && ior != kIOReturnOverrun)
	{	/* apperantly overrun is normal for string descs */
		uvcc_err("DeviceRequest", ior);
		return -1;
	}
	for(i = 0; i < (lid.bLength-2)/2; i++) buf[i] = USBToHostWord(lid.bString[i]);
	/* returned string are not null terminated */
	buf[i] = 0;
	return (lid.bLength-2)/2;
}

static void uvcc_err(const char *f,
					 int r)
{
	if(!logger) fprintf(stderr, "uvcc error: %s returned 0x%08x %s.\n", f, r, kio_return_str(r));
    else asl_log(logger, NULL, ASL_LEVEL_ERR, "%s returned 0x%08x %s.\n", f, r, kio_return_str(r));
}

/* i was lazy and just parsed IOKit/IOReturn.h and IOKit/usb/USB.h
 * TODO: fix duplicate case values, see USB.h
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
