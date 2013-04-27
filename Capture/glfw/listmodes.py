from __init__ import *

if __name__ == '__main__':
    import sys

    if not glfwInit():
        sys.exit()
    w,h,r,g,b = glfwGetDesktopMode()
    print "Desktop mode: %d x %d x %d\n" % (w,h,r+g+b)
    print "Available modes:"
    for i,mode in enumerate(glfwGetVideoModes(64)):
        w,h,r,g,b = mode
        print "%3d: %d x %d x %d" % (i,w,h,r+g+b)
    glfwTerminate()

