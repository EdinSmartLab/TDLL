##
##  High Performance Computing for Science and Engineering (HPCSE) 2018
##  TDLL: Tiny Deep Learning Library - solution code for exercises 6 and 7.
##
##  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland.
##  Distributed under the terms of the MIT license.
##
##  Created by Guido Novati (novatig@gmail.com).
##
## Dependencies: gcc>=5.4, BLAS

config ?= prod
blas ?= openblas

ifeq ($(shell uname -s), Darwin)
CXX=g++-8
LIBS += -L/usr/local/opt/openblas/lib/ -lopenblas
CXXFLAGS += -I/usr/local/opt/openblas/include/
else
CXX=g++
ifeq "$(blas)" "mkl"
CXXFLAGS+= -m64 -DUSE_MKL
LIBS+= -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core -lgomp -lpthread -lm -ldl
else
LIBS+= -lopenblas
endif
endif

CXXFLAGS+= -std=c++14 -fopenmp

ifeq "$(config)" "debug"
CXXFLAGS += -g -O0
# addressing errors generally involve accessing/writing beyond the bounds of
# the allocated memory space (i.e. seg fault)
#CXXFLAGS += -fsanitize=address
# undefined behaviors range from operation outcomes depending on uninitialized
# data to wron memory alignment:
#CXXFLAGS += -fsanitize=undefined
endif

ifeq "$(config)" "prod"
CXXFLAGS += -DNDEBUG -O3 -ffast-math
endif


CXXFLAGS+= -Wall -Wextra -Wfloat-equal -Wundef -Wcast-align -Wpedantic
CXXFLAGS+= -Wmissing-declarations -Wredundant-decls -Wshadow -Wwrite-strings
CXXFLAGS+= -Woverloaded-virtual -Wno-unused-parameter -Wno-unused-variable


exec_testGrad: main_testGrad.o
	$(CXX) $(CXXFLAGS) $(LIBS) main_testGrad.o -o $@

exec_classify: main_classify.o
	$(CXX) $(CXXFLAGS) $(LIBS) main_classify.o -o $@

exec_linear: main_linear.o
	$(CXX) $(CXXFLAGS) $(LIBS) main_linear.o -o $@

exec_nonlinear: main_nonlinear.o
	$(CXX) $(CXXFLAGS) $(LIBS) main_nonlinear.o -o $@

exec_convDeconv: main_convDeconv.o
	$(CXX) $(CXXFLAGS) $(LIBS) main_convDeconv.o -o $@

all: exec_testGrad exec_classify exec_convDeconv exec_linear exec_nonlinear
.DEFAULT_GOAL := all

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -rf *.o *.dSYM *.s *.d exec_*
