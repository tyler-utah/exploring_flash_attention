# Makefile for Flash Attention CUDA implementations

NVCC = nvcc
NVCC_FLAGS = -O3 -std=c++17 -arch=sm_80 -Xcompiler -fopenmp
LDFLAGS = -Xcompiler -fopenmp
# Adjust -arch flag based on your GPU:
# sm_70 for V100, sm_80 for A100, sm_86 for RTX 3090, sm_89 for RTX 4090

# Tuning parameters (can be overridden)
BQ ?= 8
BK ?= 8
D_TILE_QK ?= 16
D_TILE_V ?= 16
D ?= 128
THREADS_PER_BLOCK ?= 64

# Build flags
DEFINES = -DBQ=$(BQ) -DBK=$(BK) -DD_TILE_QK=$(D_TILE_QK) -DD_TILE_V=$(D_TILE_V) -DD=$(D) -DTHREADS_PER_BLOCK=$(THREADS_PER_BLOCK)

TARGET = flash_attention_v1
SOURCES = driver.cu
HEADERS = flash_attention_v1.h ../../common/standard.h

all: $(TARGET)

# Build version
$(TARGET): $(SOURCES) $(HEADERS)
	$(NVCC) $(NVCC_FLAGS) $(DEFINES) $(SOURCES) -o $(TARGET) $(LDFLAGS)

clean:
	rm -f $(TARGET)

run: $(TARGET)
	./$(TARGET)

# Print configuration
config:
	@echo "Build Configuration:"
	@echo "  BQ=$(BQ)"
	@echo "  BK=$(BK)"
	@echo "  D_TILE_QK=$(D_TILE_QK)"
	@echo "  D_TILE_V=$(D_TILE_V)"
	@echo "  D=$(D)"
	@echo "  THREADS_PER_BLOCK=$(THREADS_PER_BLOCK)"

.PHONY: all clean run config
