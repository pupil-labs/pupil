
if __name__ == '__main__':
    import subprocess as sp
    sp.call("g++ -std=c++11 -I '/usr/local/include/eigen3' -g projectionTest.cpp -o test",shell=True)
    print "BUILD COMPLETE ______________________"
    sp.call("./test",shell=True)
