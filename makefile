CC = gcc
TARGET = a.out
CFLAGS = -g -std=c99 -w -fcompare-debug-second
LIB = -lm -lmatrix -lpthread

all : $(TARGET)

libmatrix.a : libmatrix.o
	ar crv libmatrix.a libmatrix.o

.c.o : 
	$(CC) $(CFLAGS) -c -o $@ $<

$(TARGET) : main.o libmatrix.a
	$(CC) -o $@ main.o -L. $(LIB)

clean:
	rm *.o *.a *.out


