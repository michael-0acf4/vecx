ifeq ($(OS),Windows_NT)
	OUTPUT = ./bin/vecx.dll
	OUTPUT_TEST = ./bin/test.exe
	mkdir_if_not_exists = if not exist "$(1)" mkdir "$(1)"
# use of designated initializers requires '/std:c++20' on cl.exe
# nvcc (cuda_12.2.r12.2) crashes on '/std:c++20' (but works on /std:c++latest)
	CUDA_X_COMPILER = -Xcompiler="-DENABLE_CUDA_MODE /std:c++14"
else
	OUTPUT = ./bin/vecx
	OUTPUT_TEST = ./bin/test
	mkdir_if_not_exists = mkdir -p "$(1)"
	CUDA_X_COMPILER =  -Xcompiler="-Wignored-attributes -DENABLE_CUDA_MODE -fPIC -std=c++14"
endif

OPT = -O2
ifdef USE_CUDA
	CC = nvcc
	CFLAGS = $(CUDA_X_COMPILER) -I./vendors/sqlite3
	SRC_BACKEND = src/gpu.cu
else
	CC = g++
#	CFLAGS = -std=c++14 -mavx512f fPIC -I./vendors/sqlite3
	CFLAGS = -Wignored-attributes -std=c++14 -mavx2 -fPIC -I./vendors/sqlite3
	SRC_BACKEND = src/cpu.cpp
endif

build:
	@$(call mkdir_if_not_exists,bin)
	$(CC) $(CFLAGS) $(OPT) -shared -o $(OUTPUT) src/common.cpp src/vecx.cpp $(SRC_BACKEND)

test:
	@$(call mkdir_if_not_exists,bin)
	$(CC) $(CFLAGS) $(OPT) src/common.cpp $(SRC_BACKEND) src/test.cpp -o $(OUTPUT_TEST)
	$(OUTPUT_TEST)

python: build
	python e2e/basic.py

use_case: build
	python e2e/use_case.py
