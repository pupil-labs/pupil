'''
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2017  Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
'''



############# TESTS ##################
if __name__ == '__main__':

  from sys import path as syspath
  from os import path as ospath
  loc = ospath.abspath(__file__).rsplit('pupil_src', 1)
  syspath.append(ospath.join(loc[0], 'pupil_src', 'shared_modules'))
  syspath.append(ospath.join(loc[0], 'pupil_src', 'capture', 'pupil_detectors'))
  del syspath, ospath

  from collections import namedtuple
  import detector_2d
  from canny_detector import Canny_Detector
  from methods import Roi
  import cv2
  from video_capture import autoCreateCapture, CameraCaptureError,EndofVideoFileError
  import time
  from file_methods import save_object, load_object
  import shutil
  import os

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
  # detector_py.coarse_detection= False

  test_file_Folder = '../../../../../Pupil_Test_Files/' # write files to the project root folder


  def compareEllipse( ellipse_cpp , ellipse_py):
    return \
    abs(ellipse_cpp['center'][0] - ellipse_py['center'][0]) <.1 and \
    abs(ellipse_cpp['center'][1] - ellipse_py['center'][1]) <.1 and \
    abs(ellipse_cpp['major'] - ellipse_py['major'])<.1 and \
    abs(ellipse_cpp['minor'] - ellipse_py['minor'])<.1
    # abs(ellipse_cpp['angle'] - ellipse_py['angle'])<10.1

  def compare_dict(first, second):
    """ Return a dict of keys that differ with another config object.  If a value is
        not found in one fo the configs, it will be represented by KEYNOTFOUND.
        @param first:   Fist dictionary to diff.
        @param second:  Second dicationary to diff.
        @return diff:   Dict of Key => (first.val, second.val)
    """
    KEYNOTFOUNDIN1 = '<KEYNOTFOUNDIN1>'       # KeyNotFound for dictDiff
    KEYNOTFOUNDIN2 = '<KEYNOTFOUNDIN2>'       # KeyNotFound for dictDiff
    diff = {}
    sd1 = set(first)
    sd2 = set(second)
    #Keys missing in the second dict
    for key in sd1.difference(sd2):
        diff[key] = KEYNOTFOUNDIN2
    #Keys missing in the first dict
    for key in sd2.difference(sd1):
        diff[key] = KEYNOTFOUNDIN1
    #Check for differences
    for key in sd1.intersection(sd2):
        if first[key] != second[key]:
            diff[key] = (first[key], second[key])

    if diff: print diff

  def compare_contours(first,second ):
    for x,y in zip(first,second):
        for a,b in zip(x,y):
            if a[0] != b[0] or a[1] != b[1]: print "Different contours"

  def write_test_values_py():
    global test_file_Folder
    test_file_Folder += 'py/'
    #remove file path and create an empty one, befor writing new files to it
    if os.path.exists(test_file_Folder):
        shutil.rmtree(test_file_Folder)
    os.makedirs( os.path.expanduser(test_file_Folder))

    # Iterate every frame
    frameNumber = 0
    while True:
        # Get an image from the grabber
        try:
            frame = cap.get_frame()
            frameNumber += 1
        except CameraCaptureError:
            print "Capture from Camera Failed. Stopping."
            break
        except EndofVideoFileError:
            print "Video File is done."
            break
        # send to detector
        result = detector_py.detect(frame,user_roi=u_r,visualize=False)

        #save test values
        save_object( result, test_file_Folder + 'result_frame_py{}'.format(frameNumber))

        print "Frame {}".format(frameNumber)

    print "Finished writing test files py."

  def write_test_values_cpp():

    global test_file_Folder
    test_file_Folder += 'cpp/'
    #remove file path and create an empty one, befor writing new files to it
    if os.path.exists(test_file_Folder):
        shutil.rmtree(test_file_Folder)
    os.makedirs( os.path.expanduser(test_file_Folder))

    # Iterate every frame
    frameNumber = 0
    while True:
        # Get an image from the grabber
        try:
            frame = cap.get_frame()
            frameNumber += 1
        except CameraCaptureError:
            print  "Capture from Camera Failed. Stopping."
            break
        except EndofVideoFileError:
            print "Video File is done."
            break
        # send to detector
        result_cpp = detector_cpp.detect(frame, u_r,  visualize=False   )

        #save test values
        save_object( result_cpp, test_file_Folder + 'result_frame_cpp{}'.format(frameNumber))

        print "Frame {}".format(frameNumber)

    print "Finished writing test files cpp."

  def compare_test_cpp():
    global test_file_Folder
    test_file_Folder += 'cpp/'

    # Iterate every frame
    frameNumber = 0
    while True:
        # Get an image from the grabber
        try:
            frame = cap.get_frame()
            frameNumber += 1
        except CameraCaptureError:
            print "Capture from Camera Failed. Stopping."
            break
        except EndofVideoFileError:
            print "Video File is done."
            break
        # send to detector
        result = detector_cpp.detect(frame, u_r,  visualize=False  )


        #load corresponding test files
        reference_result = load_object( test_file_Folder + 'result_frame_cpp{}'.format(frameNumber))
       # reference_contours = load_object(test_file_Folder + 'contours_frame{}'.format(frameNumber))

        compare_dict(reference_result, result )
        #compare_contours( contours, reference_contours)

        print "Frame {}".format(frameNumber)

    print "Finished compare test cpp."

  def compare_test_py():

    global test_file_Folder
    test_file_Folder += 'py/'
    # Iterate every frame
    frameNumber = 0
    while True:
        # Get an image from the grabber
        try:
            frame = cap.get_frame()
            frameNumber += 1
        except CameraCaptureError:
            print "Capture from Camera Failed. Stopping."
            break
        except EndofVideoFileError:
            print "Video File is done."
            break
        # send to detector
        result = detector_py.detect(frame,user_roi=u_r,visualize=False)

        #load corresponding test files
        reference_result = load_object( test_file_Folder + 'result_frame_py{}'.format(frameNumber))

        compare_dict(reference_result, result )

        print "Frame {}".format(frameNumber)

    print "Finished compare test py."


  def compare_cpp_and_py():
    cpp_time = 0
    py_time = 0

    # Iterate every frame
    frameNumber = 0
    while True:
        # Get an image from the grabber
        try:
            frame = cap.get_frame()
            frameNumber += 1
        except CameraCaptureError:
            print "Capture from Camera Failed. Stopping."
            break
        except EndofVideoFileError:
            print"Video File is done."
            break

        frame.gray # call this here, so the conversion to gray image is not happening during timing of cpp detector
        # send to detector
        start_time = time.time()
        result_cpp = detector_cpp.detect(frame, u_r, visualize=False )
        end_time = time.time()
        cpp_time += (end_time - start_time)

        start_time = time.time()
        result_py = detector_py.detect(frame,user_roi=u_r,visualize=False)
        end_time = time.time()
        py_time += (end_time - start_time)

        # for contour in contours_py: #better way to do this ?
        #     contour.shape = (-1, 2 )

        if( 'center' in result_cpp.keys() and 'center' in result_py.keys() ):
          if not compareEllipse( result_cpp, result_py):
              print "Wrong Ellipse: cpp: {},{},{},{}  py: {},{},{},{}".format( result_cpp['center'],result_cpp['major'],result_cpp['minor'],result_cpp['angle'],  result_py['center'], result_py['major'], result_py['minor'], result_py['angle']);
        #compare_dict(reference_result, result )
        #compare_contours( contours, reference_contours)

        print "Frame {}".format(frameNumber)

    print "Finished comparing."
    print "Timing: cpp_time_average: {} , py_time_average: {}".format(cpp_time/frameNumber, py_time/frameNumber)



  #write_test_values_py()
  #compare_test_py()

  #write_test_values_cpp()
  compare_test_cpp()

  #compare_cpp_and_py()  #cpp and py values will slightly differ, because of the different implementation of the ellipse_distance function
