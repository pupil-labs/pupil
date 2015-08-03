if __name__ == '__main__':
    import subprocess as sp
    sp.call("python setup.py build_ext --inplace",shell=True)
print "BUILD COMPLETE ______________________"

import eye_model_3d
model = eye_model_3d.PyEyeModelFitter(focal_length=20)
print model
model.reset()