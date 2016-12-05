if __name__ == '__main__':
    import subprocess as sp
    sp.call("python setup_test.py build_ext --inplace",shell=True)
print "BUILD COMPLETE ______________________"
