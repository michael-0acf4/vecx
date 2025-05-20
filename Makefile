ifeq ($(OS),Windows_NT)
	OUTPUT = ./bin/vecx.dll
	OUTPUT_TEST = ./bin/test.exe
	mkdir_if_not_exists = if not exist "$(1)" mkdir "$(1)"
else
	OUTPUT = ./bin/vecx
	OUTPUT_TEST = ./bin/test
	mkdir_if_not_exists = mkdir -p "$(1)"
endif

# use of designated initializers requires '/std:c++20' on cl.exe
# nvcc (cuda_12.2.r12.2) crashes on '/std:c++20' (but works on /std:c++latest)
ifdef USE_CUDA
	CC = nvcc
	CFLAGS = -Xcompiler="-DENABLE_CUDA_MODE /std:c++14" -I./vendors/sqlite3
	SRC_BACKEND = src/gpu.cu
else
	CC = g++
	CFLAGS = -std=c++14 -mavx2 -fPIC -I./vendors/sqlite3
	SRC_BACKEND = src/cpu.cpp
endif

build:
	@$(call mkdir_if_not_exists,bin)
	$(CC) $(CFLAGS) -shared -o $(OUTPUT) src/common.cpp src/vecx.cpp $(SRC_BACKEND)

test:
	@$(call mkdir_if_not_exists,bin)
	$(CC) $(CFLAGS) src/common.cpp $(SRC_BACKEND) src/test.cpp -o $(OUTPUT_TEST)
	$(OUTPUT_TEST)

python: build
	python e2e/basic.py
