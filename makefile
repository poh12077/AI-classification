CC = gcc
TARGET_1 = train.out
TARGET_2 = test.out
CFLAGS = -g -std=c99 -w -fcompare-debug-second
LIB = -lm -lmatrix -lpthread

all : $(TARGET_1) $(TARGET_2)

libmatrix.a : libmatrix.o
	ar crv libmatrix.a libmatrix.o

.c.o : 
	$(CC) $(CFLAGS) -c -o $@ $<

$(TARGET_1) : train.o libmatrix.a
	$(CC) -o $@ train.o -L. $(LIB)

$(TARGET_2) : test.o libmatrix.a
	$(CC) -o $@ test.o -L. $(LIB)

clean:
	rm *.o *.a 


