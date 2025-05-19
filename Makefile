ifeq ($(OS),Windows_NT)
	OUTPUT = ./bin/vecx.dll
	OUTPUT_TEST = ./bin/test.exe
else
	OUTPUT = ./bin/vecx
	OUTPUT_TEST = ./bin/test
endif

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
	@if not exist bin mkdir bin
	$(CC) $(CFLAGS) -shared -o $(OUTPUT) src/common.cpp src/vecx.cpp $(SRC_BACKEND)

test:
	@if not exist bin mkdir bin
	$(CC) $(CFLAGS) src/common.cpp $(SRC_BACKEND) src/test.cpp -o $(OUTPUT_TEST)
	$(OUTPUT_TEST)

python: build
	python e2e/basic.py
