CUDA_SRC := src/main.cu
CUDA_OUT := image_convolution_cuda

HIP_SRC  := src/main.hip.cpp
HIP_OUT  := image_convolution_hip

SERIAL_SRC  := src/main.serial.c
SERIAL_OUT  := image_convolution_serial

LIBS := libs/bitmap.c libs/shared.c
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
serial: $(SERIAL_OUT)

%.o: %.c
	$(CC) -c $(FLAGS) $< -o $@

$(CUDA_OUT): $(OBJ) $(CUDA_SRC)
	$(CUDA_CC) $(FLAGS) $^ -o $@

$(HIP_OUT): $(OBJ) $(HIP_SRC)
	$(HIP_CC) $(FLAGS) $^ -o $@

$(SERIAL_OUT): $(OBJ) $(SERIAL_SRC)
	$(CC) $(FLAGS) $^ -o $@

clean:
	rm -Rf $(OBJ)
	rm -Rf $(CUDA_OUT)
	rm -Rf $(HIP_OUT)
	rm -Rf $(SERIAL_OUT)

