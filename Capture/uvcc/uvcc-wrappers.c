/* these are just wrappers.. not much to see here. */

#include "uvcc.h"

int8_t uvccScanningMode(struct uvccCam *cam, UInt8 request, int8_t value)
{
    int ret;
    int8_t pData = (request == UVC_SET_CUR ? value : 0);
    if((ret = uvccSendRequest(cam, request, UVCC_REQ_SCANNING_MODE, &pData)) != 0) return ret;
    else return pData;
};
int8_t uvccExposureMode(struct uvccCam *cam, UInt8 request, int8_t value)
{
    int ret;
    int8_t pData = (request == UVC_SET_CUR ? value : 0);
    if((ret = uvccSendRequest(cam, request, UVCC_REQ_EXPOSURE_AUTOMODE, &pData)) != 0) return ret;
    else return pData;
};
int8_t uvccExposurePrio(struct uvccCam *cam, UInt8 request, int8_t value)
{
    int ret;
    int8_t pData = (request == UVC_SET_CUR ? value : 0);
    if((ret = uvccSendRequest(cam, request, UVCC_REQ_EXPOSURE_AUTOPRIO, &pData)) != 0) return ret;
    else return pData;
};
int32_t uvccExposure(struct uvccCam *cam, UInt8 request, int32_t value)
{
    int ret;
    int32_t pData = (request == UVC_SET_CUR ? value : 0);
    if((ret = uvccSendRequest(cam, request, UVCC_REQ_EXPOSURE_ABS, &pData)) != 0) return ret;
    else return pData;
};
/* TODO: relative exposure */
int8_t uvccAutoFocus(struct uvccCam *cam, UInt8 request, int8_t value)
{
    int ret;
    int8_t pData = (request == UVC_SET_CUR ? value : 0);
    if((ret = uvccSendRequest(cam, request, UVCC_REQ_FOCUS_AUTO, &pData)) != 0) return ret;
    else return pData;
};
int16_t uvccFocus(struct uvccCam *cam, UInt8 request, int16_t value)
{
    int ret;
    int16_t pData = (request == UVC_SET_CUR ? value : 0);
    if((ret = uvccSendRequest(cam, request, UVCC_REQ_FOCUS_ABS, &pData)) != 0) return ret;
    else return pData;
};
/* TODO: relative focus */
int16_t uvccIris(struct uvccCam *cam, UInt8 request, int16_t value)
{
    int ret;
    int16_t pData = (request == UVC_SET_CUR ? value : 0);
    if((ret = uvccSendRequest(cam, request, UVCC_REQ_IRIS_ABS, &pData)) != 0) return ret;
    else return pData;
};
/* TODO: relative iris */
int16_t uvccBacklightCompensation(struct uvccCam *cam, UInt8 request, int16_t value)
{
    int ret;
    int16_t pData = (request == UVC_SET_CUR ? value : 0);
    if((ret = uvccSendRequest(cam, request, UVCC_REQ_BACKLIGHT_COMPENSATION_ABS, &pData)) != 0) return ret;
    else return pData;
};
int16_t uvccBrightness(struct uvccCam *cam, UInt8 request, int16_t value)
{
    int ret;
    int16_t pData = (request == UVC_SET_CUR ? value : 0);
    if((ret = uvccSendRequest(cam, request, UVCC_REQ_BRIGHTNESS_ABS, &pData)) != 0) return ret;
    else return pData;
};
int16_t uvccContrast(struct uvccCam *cam, UInt8 request, int16_t value)
{
    int ret;
    int16_t pData = (request == UVC_SET_CUR ? value : 0);
    if((ret = uvccSendRequest(cam, request, UVCC_REQ_CONTRAST_ABS, &pData)) != 0) return ret;
    else return pData;
};
int16_t uvccGain(struct uvccCam *cam, UInt8 request, int16_t value)
{
    int ret;
    int16_t pData = (request == UVC_SET_CUR ? value : 0);
    if((ret = uvccSendRequest(cam, request, UVCC_REQ_GAIN_ABS, &pData)) != 0) return ret;
    else return pData;
};
int8_t uvccPowerLineFrequency(struct uvccCam *cam, UInt8 request, int8_t value)
{
    int ret;
    int8_t pData = (request == UVC_SET_CUR ? value : 0);
    if((ret = uvccSendRequest(cam, request, UVCC_REQ_POWER_LINE_FREQ, &pData)) != 0) return ret;
    else return pData;
};
int16_t uvccSaturation(struct uvccCam *cam, UInt8 request, int16_t value)
{
    int ret;
    int16_t pData = (request == UVC_SET_CUR ? value : 0);
    if((ret = uvccSendRequest(cam, request, UVCC_REQ_SATURATION_ABS, &pData)) != 0) return ret;
    else return pData;
};
int16_t uvccSharpness(struct uvccCam *cam, UInt8 request, int16_t value)
{
    int ret;
    int16_t pData = (request == UVC_SET_CUR ? value : 0);
    if((ret = uvccSendRequest(cam, request, UVCC_REQ_SHARPNESS_ABS, &pData)) != 0) return ret;
    else return pData;
};
int16_t uvccGamma(struct uvccCam *cam, UInt8 request, int16_t value)
{
    int ret;
    int16_t pData = (request == UVC_SET_CUR ? value : 0);
    if((ret = uvccSendRequest(cam, request, UVCC_REQ_GAMMA_ABS, &pData)) != 0) return ret;
    else return pData;
};
int8_t uvccAutoWhiteBalanceTemp(struct uvccCam *cam, UInt8 request, int8_t value)
{
    int ret;
    int8_t pData = (request == UVC_SET_CUR ? value : 0);
    if((ret = uvccSendRequest(cam, request, UVCC_REQ_WB_TEMP_AUTO, &pData)) != 0) return ret;
    else return pData;
};
int16_t uvccWhiteBalanceTemp(struct uvccCam *cam, UInt8 request, int16_t value)
{
    int ret;
    int16_t pData = (request == UVC_SET_CUR ? value : 0);
    if((ret = uvccSendRequest(cam, request, UVCC_REQ_WB_TEMP_ABS, &pData)) != 0) return ret;
    else return pData;
};
