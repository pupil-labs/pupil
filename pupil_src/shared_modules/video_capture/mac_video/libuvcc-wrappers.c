
/* these are just wrappers.. not much to see here. */

#include "libuvcc.h"

int uvccRWScanningMode(uvccCam *cam, uint8_t request, uint8_t *value) {
	return uvccSendRequest(cam, request, UVCC_REQ_SCANNING_MODE, value);
}
int uvccRWAutoExposureMode(uvccCam *cam, uint8_t request, int8_t *value) {
	return uvccSendRequest(cam, request, UVCC_REQ_EXPOSURE_AUTOMODE, value);
}
int uvccRWAutoExposurePrio(uvccCam *cam, uint8_t request, uint8_t *value) {
	return uvccSendRequest(cam, request, UVCC_REQ_EXPOSURE_AUTOPRIO, value);
}
int uvccRWExposure(uvccCam *cam, uint8_t request, uint32_t *value) {
	return uvccSendRequest(cam, request, UVCC_REQ_EXPOSURE_ABS, value);
}
int uvccRWExposureRelative(uvccCam *cam, uint8_t request, int8_t *value) {
	return uvccSendRequest(cam, request, UVCC_REQ_EXPOSURE_REL, value);
}
/* TODO: relative exposure */
int uvccRWAutoFocus(uvccCam *cam, uint8_t request, uint8_t *value) {
	return uvccSendRequest(cam, request, UVCC_REQ_FOCUS_AUTO, value);
}
int uvccRWFocus(uvccCam *cam, uint8_t request, uint16_t *value) {
	return uvccSendRequest(cam, request, UVCC_REQ_FOCUS_ABS, value);
}
/* TODO: relative focus */
int uvccRWIris(uvccCam *cam, uint8_t request, int16_t *value) {
	return uvccSendRequest(cam, request, UVCC_REQ_IRIS_ABS, value);
}
/* TODO: relative iris */
int uvccRWBacklightCompensation(uvccCam *cam, uint8_t request, uint16_t *value) {
	return uvccSendRequest(cam, request, UVCC_REQ_BACKLIGHT_COMPENSATION_ABS, value);
}
int uvccRWBrightness(uvccCam *cam, uint8_t request, int16_t *value) {
	return uvccSendRequest(cam, request, UVCC_REQ_BRIGHTNESS_ABS, value);
}
int uvccRWContrast(uvccCam *cam, uint8_t request, uint16_t *value) {
	return uvccSendRequest(cam, request, UVCC_REQ_CONTRAST_ABS, value);
}
int uvccRWGain(uvccCam *cam, uint8_t request, uint16_t *value) {
	return uvccSendRequest(cam, request, UVCC_REQ_GAIN_ABS, value);
}
int uvccRWPowerLineFrequency(uvccCam *cam, uint8_t request, uint8_t *value) {
	return uvccSendRequest(cam, request, UVCC_REQ_POWER_LINE_FREQ, value);
}
int uvccRWAutoHue(uvccCam *cam, uint8_t request, uint8_t *value) {
	return uvccSendRequest(cam, request, UVCC_REQ_HUE_AUTO, value);
}
int uvccRWHue(uvccCam *cam, uint8_t request, int16_t *value) {
	return uvccSendRequest(cam, request, UVCC_REQ_HUE_ABS, value);
}
int uvccRWSaturation(uvccCam *cam, uint8_t request, uint16_t *value) {
	return uvccSendRequest(cam, request, UVCC_REQ_SATURATION_ABS, value);
}
int uvccRWSharpness(uvccCam *cam, uint8_t request, uint16_t *value) {
	return uvccSendRequest(cam, request, UVCC_REQ_SHARPNESS_ABS, value);
}
int uvccRWGamma(uvccCam *cam, uint8_t request, uint16_t *value) {
	return uvccSendRequest(cam, request, UVCC_REQ_GAMMA_ABS, value);
}
int uvccRWAutoWhiteBalanceTemp(uvccCam *cam, uint8_t request, uint8_t *value) {
	return uvccSendRequest(cam, request, UVCC_REQ_WB_TEMP_AUTO, value);
}
int uvccRWWhiteBalanceTemp(uvccCam *cam, uint8_t request, uint16_t *value) {
	return uvccSendRequest(cam, request, UVCC_REQ_WB_TEMP_ABS, value);
}
int uvccRWAutoWhiteBalanceComponent(uvccCam *cam, uint8_t request, uint8_t *value) {
	return uvccSendRequest(cam, request, UVCC_REQ_WB_COMPONENT_AUTO, value);
}
int uvccRWWhiteBalanceComponent(uvccCam *cam, uint8_t request, uint16_t *blue, uint16_t *red) {
	int32_t value = ((*red) << 16) | (*blue);
	int ret = uvccSendRequest(cam, request, UVCC_REQ_WB_COMPONENT_ABS, (void *)&value);
	*red = value >> 16;
	*blue = *((uint16_t *)&value);
	return ret;
}
