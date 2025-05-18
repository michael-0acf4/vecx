NAME = vecx
# CC = nvcc
CC = gcc
CFLAGS = -fPIC -shared -I./vendors/sqlite3
# CFLAGS = -shared -I./vendors/sqlite3
SRC = src/$(NAME).c

build:
	$(CC) $(CFLAGS) -o $(NAME).dll $(SRC)

test: build
	python e2e/basic.py
