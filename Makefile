CUDA_SRC := src/main.cu
CUDA_OUT := image_convolution_cuda

HIP_SRC  := src/main.hip.cpp
HIP_OUT  := image_convolution_hip

OPENCL_SRC := src/main.opencl.cpp
OPENCL_OBJ := src/main.opencl.o
OPENCL_OUT := image_convolution_opencl
OPENCL_DFLAGS := -DCL_TARGET_OPENCL_VERSION=220
OPENCL_IFLAGS := -I/opt/rocm-4.2.0/opencl/include
OPENCL_LFLAGS := -lOpenCL -L/opt/rocm-4.2.0/lib

SERIAL_SRC  := src/main.serial.c
SERIAL_OUT  := image_convolution_serial

LIBS := libs/bitmap.c libs/shared.c
LIB_OBJ := $(patsubst %.c,%.o,$(LIBS))

CC      := gcc
CUDA_CC := nvcc
HIP_CC  := hipcc

ifdef DEBUG
CFLAGS := -g -O0 -DDEBUG
else
CFLAGS := -O2
endif

.PHONY: clean

hip: $(HIP_OUT)
cuda: $(CUDA_OUT)
opencl: $(OPENCL_OUT)
serial: $(SERIAL_OUT)


$(CUDA_OUT): $(LIB_OBJ) $(CUDA_SRC) 
	$(CUDA_CC) $(CFLAGS) $^ -o $@

$(HIP_OUT): $(LIB_OBJ) $(HIP_SRC) 
	$(HIP_CC) $(CFLAGS) $^ -o $@

$(OPENCL_OBJ): $(OPENCL_SRC)
	$(CC) -c $(CFLAGS) $(OPENCL_DFLAGS) $(OPENCL_IFLAGS) $^ -o $@

$(OPENCL_OUT): $(LIB_OBJ) $(OPENCL_OBJ)
	$(CC) $^ $(OPENCL_LFLAGS) -o $@

$(SERIAL_OUT): $(SERIAL_SRC) $(LIB_OBJ)
	$(CC) $(CFLAGS) $^ -o $@

%.o: %.c
	$(CC) -c $(CFLAGS) $< -o $@

clean:
	rm -Rf $(LIB_OBJ) $(OPENCL_OBJ)  $(CUDA_OUT) $(HIP_OUT) $(SERIAL_OUT) $(OPENCL_OUT)

