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
  import cv2
  from video_capture import autoCreateCapture, CameraCaptureError,EndofVideoFileError
  import time

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
      print "Could not retrieve image from capture"
      cap.close()

  Pool = namedtuple('Pool', 'user_dir');
  pool = Pool('/')
  u_r = Roi(frame.img.shape)

  #Our detectors we wanna compare
  detector_cpp = detector_2d.Detector_2D()
  detector_py = Canny_Detector(pool)


  def compareEllipse( ellipse_cpp , ellipse_py):
    return ellipse_cpp['center'] == ellipse_py['center'] and \
    ellipse_cpp['major'] == ellipse_py['major'] and \
    ellipse_cpp['minor'] == ellipse_py['minor'] and \
    ellipse_cpp['angle'] == ellipse_py['angle']


  cpp_time = 0
  py_time = 0

  # Iterate every frame
  frameNumber = 0
  while True:
      # Get an image from the grabber
      try:
          frame = cap.get_frame_nowait()
          frameNumber += 1
      except CameraCaptureError:
          print "Capture from Camera Failed. Stopping."
          break
      except EndofVideoFileError:
          print"Video File is done."
          break
      # send to detector
      start_time = time.time()
      result_cpp = detector_cpp.detect(frame, u_r, None )
      end_time = time.time()
      cpp_time += (end_time - start_time)

      start_time = time.time()
      result_py,contours_py = detector_py.detect(frame,user_roi=u_r,visualize=False)
      end_time = time.time()
      py_time += (end_time - start_time)

      for contour in contours_py: #better way to do this ?
          contour.shape = (-1, 2 )

      if( 'center' in result_cpp.keys() and 'center' in result_py.keys() ):
        if not compareEllipse( result_cpp, result_py):
            print "Wrong Ellipse: cpp: {},{},{},{}  py: {},{},{},{}".format( result_cpp['center'],result_cpp['major'],result_cpp['minor'],result_cpp['angle'],  result_py['center'], result_py['major'], result_py['minor'], result_py['angle']);
      #compare_dict(reference_result, result )
      #compare_contours( contours, reference_contours)

      print "Frame {}".format(frameNumber)

  print "Finished Testing."
  print "Timing: cpp_time_average: {} , py_time_average: {}".format(cpp_time/frameNumber, py_time/frameNumber)


