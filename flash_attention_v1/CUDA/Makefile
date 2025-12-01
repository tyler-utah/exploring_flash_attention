# Makefile for Flash Attention CUDA implementations

NVCC = nvcc
NVCC_FLAGS = -O3 -std=c++17 -arch=sm_80 -Xcompiler -fopenmp
LDFLAGS = -Xcompiler -fopenmp
# Adjust -arch flag based on your GPU:
# sm_70 for V100, sm_80 for A100, sm_86 for RTX 3090, sm_89 for RTX 4090

TARGET_V1 = flash_attention_v1
TARGET_OPT1 = flash_attention_v1_opt1
SOURCES = driver.cu
HEADERS = standard.h load_shared_memory.h

all: $(TARGET_V1) $(TARGET_OPT1)

# Build baseline version (without USE_OPT1 flag)
$(TARGET_V1): $(SOURCES) $(HEADERS) flash_attention_v1.h
	$(NVCC) $(NVCC_FLAGS) $(SOURCES) -o $(TARGET_V1) $(LDFLAGS)

# Build opt1 version (with USE_OPT1 flag)
$(TARGET_OPT1): $(SOURCES) $(HEADERS) flash_attention_v1_opt1.h
	$(NVCC) $(NVCC_FLAGS) -DUSE_OPT1=1 $(SOURCES) -o $(TARGET_OPT1) $(LDFLAGS)

clean:
	rm -f $(TARGET_V1) $(TARGET_OPT1)

run_v1: $(TARGET_V1)
	./$(TARGET_V1)

run_opt1: $(TARGET_OPT1)
	./$(TARGET_OPT1)

.PHONY: all clean run_v1 run_opt1
