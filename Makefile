NAME = vecx

# make USE_CUDA=1
ifdef USE_CUDA
	CC = nvcc
	CFLAGS = -Xcompiler="-DENABLE_CUDA_MODE /std:c++14" -shared -I./vendors/sqlite3
	SRC = src/common.cpp  src/gpu.cu src/$(NAME).cpp
else
	CC = g++
	CFLAGS = -std=c++14 -fPIC -shared -I./vendors/sqlite3
	SRC = src/$(NAME).cpp src/common.cpp src/cpu.cpp
endif

build:
	$(CC) $(CFLAGS) -o $(NAME).dll $(SRC)

test: build
	python e2e/basic.py
