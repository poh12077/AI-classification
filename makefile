CC = gcc
TARGET_1 = train.out
TARGET_2 = test.out
CFLAGS = -g -std=c99 -w -fcompare-debug-second
LIB = -lm -ldeeplearning 

all : $(TARGET_1) $(TARGET_2)

libdeeplearning.a : libdeeplearning.o
	ar crv libdeeplearning.a libdeeplearning.o

.c.o : 
	$(CC) $(CFLAGS) -c -o $@ $<

$(TARGET_1) : train.o libdeeplearning.a
	$(CC) -o $@ train.o -L. $(LIB)

$(TARGET_2) : test.o libdeeplearning.a
	$(CC) -o $@ test.o -L. $(LIB)

clean:
	rm *.o *.a 


