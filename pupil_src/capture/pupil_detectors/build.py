def build_cpp_extension():
    import subprocess as sp
    import os
    src_loc = os.path.dirname(os.path.realpath(__file__))
    cwd = os.getcwd()
    sp.call("cd %s && python setup.py build_ext --inplace && cd %s"%(src_loc,cwd),shell=True)

if __name__ == '__main__':
    build_cpp_extension()

