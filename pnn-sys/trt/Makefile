OUTNAME_RELEASE = trt
OUTNAME_DEBUG   = trt_debug

INCLUDE_DIR=/opt/cuda/include
LIB_DIR=/opt/cuda/lib64
LIBS=-lnvinfer -lnvinfer_plugin -lcuda -lcudart

CPP=g++
CFLAGS=-Wall -I$(INCLUDE_DIR) -L$(LIB_DIR)
NVCC=nvcc
src=trt.h activation.hpp main.cpp activation.cpp utils.hpp builder.hpp builder.cpp utils.cpp engine.cpp engine.hpp trt.cpp

all: build clean

build:
	@$(NVCC) -c mish.cu --use_fast_math -arch=sm_80 -I$(INCLUDE_DIR) -cudart=shared -lcudart -o mish.o
	@$(CPP) $(src) mish.o $(CFLAGS) $(LIBS) -o trt

clean:
	@rm mish.o