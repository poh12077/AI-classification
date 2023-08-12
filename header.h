#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdlib.h>



#define inputNode 784
#define node_0 10
#define node_1 10
#define outputNode 10
#define learningRateMacro 0.1 //traing work
#define softmaxBase 1.001
#define iteration 1
#define parameterFilePath "parameter"

#define trainBatch 1000
#define trainDataPath "../data/MNIST_CSV/mnist_train.csv"
#define trainDataSize 60000

#define testBatch 1000
#define testDataPath "../data/MNIST_CSV/mnist_test.csv"
#define testDataSize 10000

typedef struct Data{
    double input[inputNode];
    double output[outputNode];
}Data;

typedef struct Gradient{
    double* w;
    double* b;
}Gradient;

typedef struct HiddenLayer{
    double _0[node_0];
    double _1[node_1];
    double _2[outputNode];
}HiddenLayer;

typedef struct Weight{
    double _0[node_0][inputNode];
    double _1[node_1][node_0];
    double _2[outputNode][node_1];
}W;

typedef struct Bias{
    double _0[node_0];
    double _1[node_1];
    double _2[outputNode];
}B;

void predict( W* w, B* b, Data *data, HiddenLayer *hiddenLayer, int* numberOfCorrect );

double GetDistance(double *realValue, double *predictedValue);
void forward(W* w, B* b, Data* data, HiddenLayer* hiddenLayer);
void initParameter( double* w, unsigned int row, unsigned int column);
double getRandom();
Gradient getGradient(W* w, B* b, Data *data, HiddenLayer *hiddenLayer, double h );
double getLoss(W* w, B* b, Data *data, HiddenLayer *hiddenLayer);
void gradientDescent(W* w, B* b, Data* data, HiddenLayer* hiddenLayer, double learningRate );
void normalize( double* data, int size, double max );
void readMnist(const char* path, double* mnist, unsigned int data_num );
int convertCharToInt( unsigned char x);
void oneHotEncoding(int x, double* arr, int size);
void writeParameter(char* path, W* w, B* b );
void readParameter(char* path, W* w, B* b );





void originalSoftmax(double* vector, int size);
void sigmoid(double* vector, int size);
void fixedSoftmax(double* vector, int size);
double absolute(double x);
double getSqrt( double x );
double getNorm( double *v, unsigned size );
void multiplyMatrices(double *a_, double *b_, unsigned int a_row, unsigned int a_column, unsigned int b_column, double *result_  );
void addMatrices(double *a, double *b, double *result, unsigned int row, unsigned int column );
void subtractMatrices(double *a, double *b, double *result, unsigned int row, unsigned int column );
int findBiggestOrder(double* arr, int size);
