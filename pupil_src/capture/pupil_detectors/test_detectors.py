if __name__ == '__main__':
    import subprocess as sp
    sp.call("python setup.py build_ext --inplace",shell=True)
print "BUILD COMPLETE ______________________"


############# TESTS ##################
if __name__ == '__main__':

  from sys import path as syspath
  from os import path as ospath
  loc = ospath.abspath(__file__).rsplit('pupil_src', 1)
  syspath.append(ospath.join(loc[0], 'pupil_src', 'shared_modules'))
  del syspath, ospath

  from collections import namedtuple
  import detector_2d
  from canny_detector import Canny_Detector
  from methods import Roi
  import timeit
  import cv2
  from video_capture import autoCreateCapture, CameraCaptureError,EndofVideoFileError
  #write test cases
  cap_src = '/Users/patrickfuerst/Documents/Projects/Pupil-Laps/recordings/2015_09_16/000/eye0.mp4'
  cap_size = (640,480)

  # Initialize capture
  cap = autoCreateCapture(cap_src, timebase=None)
  default_settings = {'frame_size':cap_size,'frame_rate':30}
  cap.settings = default_settings
  # Test capture
  try:
      frame = cap.get_frame()
  except CameraCaptureError:
      logger.error("Could not retrieve image from capture")
      cap.close()

  Pool = namedtuple('Pool', 'user_dir');
  pool = Pool('/')
  u_r = Roi(frame.img.shape)

  #Our detectors we wanna compare
  detector_cpp = detector_2d.Detector_2D()
  detector_py = Canny_Detector(pool)


  def compareEllipse( ellipse_cpp , ellipse_py):
    return ellipse_cpp == ellipse_py




  # Iterate every frame
  frameNumber = 0
  while True:
      # Get an image from the grabber
      try:
          frame = cap.get_frame()
          frameNumber += 1
      except CameraCaptureError:
          logger.error("Capture from Camera Failed. Stopping.")
          break
      except EndofVideoFileError:
          logger.warning("Video File is done.")
          break
      # send to detector
      result_cpp = detector_cpp.detect(frame, u_r, None )
      result_py,contours_py = detector_py.detect(frame,user_roi=u_r,visualize=False)

      for contour in contours_py: #better way to do this ?
          contour.shape = (-1, 2 )

      if 'ellipse' in result_cpp.keys() and  'ellipse' in result_py.keys() : #early exits don't have this
        if not compareEllipse( result_cpp['ellipse'], result_py['ellipse']):
            print "Wrong Ellipse: cpp: {}  py: {}".format( result_cpp['ellipse'],  result_py['ellipse']);
      #compare_dict(reference_result, result )
      #compare_contours( contours, reference_contours)

      print "Frame {}".format(frameNumber)

  print "Finished Testing."





