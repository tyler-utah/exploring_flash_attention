# Makefile for Flash Attention CUDA implementations

NVCC = nvcc
NVCC_FLAGS = -O3 -std=c++17 -arch=sm_80 -Xcompiler -fopenmp
LDFLAGS = -Xcompiler -fopenmp
# Adjust -arch flag based on your GPU:
# sm_70 for V100, sm_80 for A100, sm_86 for RTX 3090, sm_89 for RTX 4090

# Tuning parameters (can be overridden)
BQ ?= 16
BK ?= 16
D_TILE_QK ?= 32
D_TILE_V ?= 32
D ?= 128
THREADS_PER_BLOCK ?= 256

# Build flags
DEFINES = -DBQ=$(BQ) -DBK=$(BK) -DD_TILE_QK=$(D_TILE_QK) -DD_TILE_V=$(D_TILE_V) -DD=$(D) -DTHREADS_PER_BLOCK=$(THREADS_PER_BLOCK)

TARGET_V1 = flash_attention_v1
TARGET_OPT = flash_attention_v1_opt
SOURCES = driver.cu
HEADERS = flash_attention_v1.h flash_attention_v1_opt.h ../../common/standard.h

all: $(TARGET_V1) $(TARGET_OPT)

# Build baseline version
$(TARGET_V1): $(SOURCES) $(HEADERS)
	$(NVCC) $(NVCC_FLAGS) $(DEFINES) $(SOURCES) -o $(TARGET_V1) $(LDFLAGS)

# Build opt version (with USE_OPT flag)
$(TARGET_OPT): $(SOURCES) $(HEADERS)
	$(NVCC) $(NVCC_FLAGS) $(DEFINES) -DUSE_OPT=1 $(SOURCES) -o $(TARGET_OPT) $(LDFLAGS)

clean:
	rm -f $(TARGET_V1) $(TARGET_OPT)

run_v1: $(TARGET_V1)
	./$(TARGET_V1)

run_opt: $(TARGET_OPT)
	./$(TARGET_OPT)
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

.PHONY: all clean run_v1 run_opt config
