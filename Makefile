CUDA_SRC := src/main.cu
CUDA_OUT := image_convolution_cuda

HIP_SRC  := src/main.hip.cpp
HIP_OUT  := image_convolution_hip

LIBS := $(wildcard libs/*.c)
OBJ := $(patsubst %.c,%.o,$(LIBS))

CC      := gcc
CUDA_CC := nvcc
HIP_CC  := hipcc

ifdef DEBUG
FLAGS := -g -O0
else
FLAGS := -O2
endif

.PHONY: clean

hip: $(HIP_OUT)
cuda: $(CUDA_OUT)

$(OBJ): $(LIBS)
	$(CC) -c $(FLAGS) $< -o $@

$(CUDA_OUT): $(OBJ) $(CUDA_SRC)
	$(CUDA_CC) $(FLAGS) $^ -o $@

$(HIP_OUT): $(OBJ) $(HIP_SRC)
	$(HIP_CC) $(FLAGS) $^ -o $@

clean:
	rm -Rf $(CUDA_OUT)
	rm -Rf $(HIP_OUT)

