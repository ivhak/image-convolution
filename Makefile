BIN_BASENAME := image-convolution

CUDA_SRC := src/cuda/main.cu
CUDA_BIN := $(BIN_BASENAME)-cuda

HIP_SRC  := src/hip/main.hip.cpp
HIP_OBJ  := src/hip/main.hip.o
HIP_BIN  := $(BIN_BASENAME)-hip
HIP_NVIDIA_BIN  := $(BIN_BASENAME)-hip-nvidia

OPENCL_SRC := src/opencl/main.cpp
OPENCL_BIN := $(BIN_BASENAME)-opencl
OPENCL_DFLAGS := -DCL_TARGET_OPENCL_VERSION=220
OPENCL_IFLAGS := -I/opt/rocm-4.2.0/opencl/include
OPENCL_LFLAGS := -lOpenCL -L/opt/rocm-4.2.0/lib

SERIAL_SRC  := src/serial/main.c
SERIAL_BIN  := $(BIN_BASENAME)-serial

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

ifdef VERBOSE
DFLAGS += -DVERBOSE
endif

.PHONY: clean

hip:    $(HIP_BIN)
cuda:   $(CUDA_BIN)
opencl: $(OPENCL_BIN)
serial: $(SERIAL_BIN)

hip-nvidia: export HIP_PLATFORM=nvidia
hip-nvidia: export CUDA_PATH=/cm/shared/apps/cuda11.0/toolkit/11.0.3
hip-nvidia: CPPFLAGS += -arch=sm_70
hip-nvidia: $(HIP_NVIDIA_BIN)

tools:  bmpdiff bmptile

bmpdiff: $(LIB_OBJ) src/tools/bmpdiff.c
	$(CC) $(CFLAGS) $^ -o $@

bmptile: $(LIB_OBJ) src/tools/bmptile.c
	$(CC) $(CFLAGS) $^ -o $@

$(CUDA_BIN): $(LIB_OBJ) $(CUDA_SRC)
	$(CUDA_CC) $(CFLAGS) $(DFLAGS) $^ -o $@

$(HIP_OBJ): $(HIP_SRC)
	$(HIP_CC) -c $(CFLAGS) $(DFLAGS) $^ -o $@

$(HIP_BIN): $(LIB_OBJ) $(HIP_OBJ)
	$(HIP_CC) $(CFLAGS) $(DFLAGS) $^ -o $@

$(HIP_NVIDIA_BIN): $(LIB_OBJ) $(HIP_OBJ)
	$(HIP_CC) $(CFLAGS) $(DFLAGS) $^ -o $@

$(OPENCL_BIN): $(LIB_OBJ) $(OPENCL_SRC)
	$(CC) $(CFLAGS) $(DFLAGS) $(OPENCL_DFLAGS) $(OPENCL_IFLAGS) $^ $(OPENCL_LFLAGS) -o $@

$(SERIAL_BIN): $(SERIAL_SRC) $(LIB_OBJ)
	$(CC) $(CFLAGS) $^ -o $@

%.o: %.c
	$(CC) -c $(CFLAGS) $< -o $@

clean:
	rm -Rf $(LIB_OBJ) $(CUDA_BIN) $(HIP_BIN) $(HIP_OBJ) $(HIP_NVIDIA_BIN) $(SERIAL_BIN) $(OPENCL_BIN) bmpdiff bmptile

