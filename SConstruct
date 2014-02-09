import os
import sys

SRC = Split('''
            LLC_Encoder.cpp
            main.cpp
            ''')


INCLUDE_PATH = Split('''./include
                        /usr/include
                        /usr/local/include/
                        /opt/intel/mkl/include/
                        ''')

LIB_PATH = Split('''./lib
                    /usr/local/lib
                    /usr/lib
                    /opt/intel/mkl/lib/intel64/
                    ''')

_LIBS = Split('''
                 libvl.so
                 libmkl_rt.so
                ''')


env = Environment(LIBPATH=LIB_PATH, LIBS=_LIBS, CPPPATH=INCLUDE_PATH, LINKFLAGS='-fopenmp',
                  CFLAGS='-O3', CXXFLAGS='-O3', CXX='g++');

env.StaticLibrary(target='LLC_Encoder', source=SRC)
env.Program(target='LLC_Encoder.bin', source=SRC)
