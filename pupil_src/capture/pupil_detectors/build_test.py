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

  import detector_2d
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

  detector = detector_2d.Detector_2D()
  u_r = Roi(frame.img.shape)

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
      result = detector.detect(frame, u_r, None )

      #cv2.imshow('Color',frame.img)
      #cv2.imshow('Gray',frame.gray)
      #cv2.waitKey(1)
      #print result

     # print "Frame {}".format(frameNumber)

  print "Finished writing Test file."





