CUDA_SRC := src/hip/main.cu
CUDA_OUT := image_convolution_cuda

HIP_SRC  := src/hip/main.hip.cpp
HIP_OUT  := image_convolution_hip

OPENCL_SRC := src/opencl/main.cpp
OPENCL_OBJ := src/opencl/main.l.o
OPENCL_OUT := image_convolution_opencl
OPENCL_DFLAGS := -DCL_TARGET_OPENCL_VERSION=220
OPENCL_IFLAGS := -I/opt/rocm-4.2.0/opencl/include
OPENCL_LFLAGS := -lOpenCL -L/opt/rocm-4.2.0/lib

SERIAL_SRC  := src/serial/main.c
SERIAL_OUT  := image_convolution_serial

LIBS := lib/bitmap.c lib/shared.c
LIB_OBJ := $(patsubst %.c,%.o,$(LIBS))

CC      := gcc
CUDA_CC := nvcc
HIP_CC  := hipcc

ifdef DEBUG
CFLAGS := -g -O0 -DDEBUG
else
CFLAGS := -O2
endif

# Turn of shared memory in the Cuda, HIP and OpenCl implementations
ifdef SHARED_MEM
DFLAGS += -DSHARED_MEM
endif

ifdef VERBOSE
DFLAGS += -DVERBOSE
endif

.PHONY: clean

hip: $(HIP_OUT)
cuda: $(CUDA_OUT)
opencl: $(OPENCL_OUT)
serial: $(SERIAL_OUT)

bmpdiff: $(LIB_OBJ) src/bmpdiff.c
	$(CC) $(CFLAGS) $^ -o $@

$(CUDA_OUT): $(LIB_OBJ) $(CUDA_SRC)
	$(CUDA_CC) $(CFLAGS) $(DFLAGS) $^ -o $@

$(HIP_OUT): $(LIB_OBJ) $(HIP_SRC)
	$(HIP_CC) $(CFLAGS) $(DFLAGS) $^ -o $@

$(OPENCL_OBJ): $(OPENCL_SRC)
	$(CC) -c $(CFLAGS) $(DFLAGS) $(OPENCL_DFLAGS) $(OPENCL_IFLAGS) $^ -o $@

$(OPENCL_OUT): $(LIB_OBJ) $(OPENCL_OBJ)
	$(CC) $^ $(OPENCL_LFLAGS) -o $@

$(SERIAL_OUT): $(SERIAL_SRC) $(LIB_OBJ)
	$(CC) $(CFLAGS) $^ -o $@

%.o: %.c
	$(CC) -c $(CFLAGS) $< -o $@

clean:
	rm -Rf $(LIB_OBJ) $(OPENCL_OBJ)  $(CUDA_OUT) $(HIP_OUT) $(SERIAL_OUT) $(OPENCL_OUT) bmpdiff

