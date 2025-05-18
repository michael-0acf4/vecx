NAME = add
CC = gcc
CFLAGS = -fPIC -shared -I./sqlite3
SRC = src/main.c

build:
	$(CC) $(CFLAGS) -o $(NAME).dll $(SRC)
