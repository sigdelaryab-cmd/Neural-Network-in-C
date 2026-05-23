CC = gcc
CFLAGS = -O3 -Wall -Iinclude
OMPFLAGS = -fopenmp
LIBS = -lm

SRC = src/main.c src/data.c src/nn.c src/train.c src/utils.c
OBJ = $(SRC:.c=.o)

TARGET = nn

all: $(TARGET)

$(TARGET): $(SRC)
    $(CC) $(CFLAGS) $(OMPFLAGS) $(SRC) $(LIBS) -o $(TARGET)

serial:
    $(CC) $(CFLAGS) $(SRC) $(LIBS) -o nn_serial

openmp:
    $(CC) $(CFLAGS) $(OMPFLAGS) $(SRC) $(LIBS) -o nn_openmp

clean:
    rm -f $(TARGET) nn_serial nn_openmp src/*.o