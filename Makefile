CUDA_SRC := src/cuda/main.cu
CUDA_OUT := image_convolution_cuda

HIP_SRC  := src/hip/main.hip.cpp
HIP_OUT  := image_convolution_hip

OPENCL_SRC := src/opencl/main.cpp
OPENCL_OUT := image_convolution_opencl
OPENCL_DFLAGS := -DCL_TARGET_OPENCL_VERSION=220
OPENCL_IFLAGS := -I/opt/rocm-4.2.0/opencl/include
OPENCL_LFLAGS := -lOpenCL -L/opt/rocm-4.2.0/lib

SERIAL_SRC  := src/serial/main.c
SERIAL_OUT  := image_convolution_serial

LIBS := lib/bitmap.c lib/shared.c lib/filter.c
LIB_OBJ := $(patsubst %.c,%.o,$(LIBS))

CC      := gcc
CUDA_CC := nvcc
HIP_CC  := hipcc


DFLAGS :=
CFLAGS :=
ifdef DEBUG
CFLAGS += -g -O0
DFLAGS += -DDEBUG
else
CFLAGS += -O3
endif

# Use shared memory in the Cuda, HIP and OpenCl implementations
# ifdef SHARED_MEM
# DFLAGS += -DSHARED_MEM
# endif

ifdef VERBOSE
DFLAGS += -DVERBOSE
endif

.PHONY: clean

hip:    $(HIP_OUT)
cuda:   $(CUDA_OUT)
opencl: $(OPENCL_OUT)
serial: $(SERIAL_OUT)
tools:  bmpdiff bmptile

bmpdiff: $(LIB_OBJ) src/tools/bmpdiff.c
	$(CC) $(CFLAGS) $^ -o $@

bmptile: $(LIB_OBJ) src/tools/bmptile.c
	$(CC) $(CFLAGS) $^ -o $@

$(CUDA_OUT): $(LIB_OBJ) $(CUDA_SRC)
	$(CUDA_CC) $(CFLAGS) $(DFLAGS) $^ -o $@

$(HIP_OUT): $(LIB_OBJ) $(HIP_SRC)
	$(HIP_CC) $(CFLAGS) $(DFLAGS) $^ -o $@

$(OPENCL_OUT): $(LIB_OBJ) $(OPENCL_SRC)
	$(CC) $(CFLAGS) $(DFLAGS) $(OPENCL_DFLAGS) $(OPENCL_IFLAGS) $^ $(OPENCL_LFLAGS) -o $@

$(SERIAL_OUT): $(SERIAL_SRC) $(LIB_OBJ)
	$(CC) $(CFLAGS) $^ -o $@

%.o: %.c
	$(CC) -c $(CFLAGS) $< -o $@

clean:
	rm -Rf $(LIB_OBJ) $(CUDA_OUT) $(HIP_OUT) $(SERIAL_OUT) $(OPENCL_OUT) bmpdiff bmptile

