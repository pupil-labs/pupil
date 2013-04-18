"""
a code snippit that enumerates all VideoCapture devices attached
"""
import QTKit

# QTCaptureDevice inputDevicesWithMediaType:QTMediaTypeVideo
qt_cameras =  QTKit.QTCaptureDevice.inputDevicesWithMediaType_(QTKit.QTMediaTypeVideo)
for i,q in enumerate(qt_cameras):
    Quicktime_uId = q.uniqueID()
    OpenCv_id = i
    name = q.localizedDisplayName().encode('utf-8')
    print name
    print OpenCv_id