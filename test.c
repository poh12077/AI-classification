#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdlib.h>
#include "header.h"
#include <pthread.h>

#define resolution 512*512
#define batch 784
#define node_0 10
#define node_1 10
#define node_2 10
#define outputNode 10
#define learningRateMacro 1 //traing work 
//#define learningRateMacro 0.5  
#define base 1.001
#define iteration 1
#define testDataNum 1000
#define parameterFilePath "parameter"
#define dataPath "../data/MNIST_CSV/mnist_test.csv"
#define mnistDataNum 10000

typedef struct Data{
	double input[batch];
	double output[outputNode];
}Data;

typedef struct Gradient{
	double* w;
	double* b;
}Gradient;

typedef struct HiddenLayer{
	double _0[node_0];
	double _1[node_1];
	double _2[node_2];
}HiddenLayer;

typedef struct Weight{
	double _0[node_0][batch];
	double _1[node_1][node_0];
	double _2[node_2][node_1];
}W;

typedef struct Bias{
	double _0[node_0];
	double _1[node_1];
	double _2[node_2];
}B;

typedef struct GradientDescentInput{
	W* w;
	B* b;
	Data* data;
	HiddenLayer* hiddenLayer;
	double learningRate;
}GradientDescentInput;


double GetDistance(double *realValue, double *predictedValue);
void predict( W* w, B* b, Data *data, HiddenLayer *hiddenLayer, int* numberOfCorrect );
void forward(W* w, B* b, Data* data, HiddenLayer* hiddenLayer);
void initParameter( double* w, unsigned int row, unsigned int column);
double getRandom();
Gradient getGradient(W* w, B* b, Data *data, HiddenLayer *hiddenLayer, double h );
double getLoss(W* w, B* b, Data *data, HiddenLayer *hiddenLayer);
void gradientDescent(W* w, B* b, Data* data, HiddenLayer* hiddenLayer, double learningRate );
void runRandomDataTest(int trials, W* w, B* b, HiddenLayer *hiddenLayer );
void* runTraining(void *input);
void normalize( double* data, int size, double max );
void readImage(const char* path, Data* data, int isLena );
void readMnist(const char* path, double* mnist, unsigned int data_num );
int convertCharToInt( unsigned char x);
void oneHotEncoding(int x, double* arr, int size);
void writeParameter(char* path, W* w, B* b );
void readParameter(char* path, W* w, B* b );

pthread_mutex_t key = PTHREAD_MUTEX_INITIALIZER;

Data testData[testDataNum];
double mnist[ mnistDataNum ][28*28+1];
	
int main()
{

	srand(time(NULL));

	readMnist( dataPath , mnist, mnistDataNum );

	for(int i=0;i<testDataNum;i++){
		int random = rand() % mnistDataNum;
		memcpy( testData[i].input, &mnist[ random ][1], sizeof(double)*batch );   	
		normalize( testData[i].input, batch, 255 );
		oneHotEncoding( mnist[ random ][0], testData[i].output , outputNode);
	}

	HiddenLayer hiddenLayer;
	W w;
	B b;

	readParameter( parameterFilePath, &w, &b );

	int numberOfCorrect=0;
	printf("prediction start\n\n");
	for(int i=0;i<testDataNum; i++){
		predict( &w, &b, testData+i, &hiddenLayer, &numberOfCorrect );
		printf("\n");
	}
	double accuracy = (double)numberOfCorrect / (double)testDataNum ;
	accuracy *= 100;
	printf("accuracy : %lf\n", accuracy );

	return 0;
}

void readParameter(char* path, W* w, B* b ){
	FILE *fptr;
	fptr = fopen(path,"r");
	if(fptr == NULL)
	{
	  printf("[readParameter Error]\n");
	}

	fread( w, sizeof(W), 1, fptr);
	fread( b, sizeof(B), 1, fptr);

	fclose(fptr);
}

void oneHotEncoding(int x, double* arr, int size){
	for(int i=0; i< size; i++){
		if( i==x){
			arr[i]=1;
		}else{
			arr[i]=0;
		}
	}
}

int convertCharToInt( unsigned char x){
	switch ( x ){
		case '0':
			return 0;
		case '1':
			return 1;
		case '2':
			return 2;
		case '3':
			return 3;
		case '4':
			return 4;
		case '5':
			return 5;
		case '6':
			return 6;
		case '7':
			return 7;
		case '8':
			return 8;
		case '9':
			return 9;
	}
}

void readMnist(const char* path, double* mnist, unsigned int data_num ){
	FILE *fptr;
	if ((fptr = fopen( path ,"rb")) == NULL){
		printf("Error! opening file");
		exit(1);
	}

	unsigned int n=0;
	int sum=0;
	int j=0;
	int count=0;
	while(1){
		unsigned char x =(unsigned char)fgetc( fptr );
		if( x == '\n' ){
			mnist[n]=sum;
			n++;
			j=0;
			sum=0;
			count++;
			if( count == data_num){
				break;
			}
			continue;
		}
		if( x != ',' ){
			if(j!=0){
				sum *= 10;
				sum+= convertCharToInt(x) ;
			}else{
				sum+=  convertCharToInt(x);
				j++;
			}
		}else{
			mnist[n]=sum;
			n++;
			j=0;
			sum=0;
		}
	}

	fclose(fptr); 
}

void normalize( double* data, int size, double max ){
	for( int i=0;i<size;i++){
		data[i] = data[i]/max;
	}
}

void predict( W* w, B* b, Data *data, HiddenLayer *hiddenLayer, int* numberOfCorrect ){
	forward(w, b, data, hiddenLayer );
	double loss = absolute( getLoss( w, b, data, hiddenLayer ) );
	int label = findBiggestOrder( data->output, outputNode );
	int prediction = findBiggestOrder( hiddenLayer->_2, outputNode );

	if( label == prediction ){
		printf("correct!! label is %d\n", label);
		*numberOfCorrect = (*numberOfCorrect) + 1;
	}else{
		printf("wrong!! label is %d\n", label);
	}

	printf("lose : %lf ", loss );
	printf("\n");

	for(int k=0;k<outputNode;k++){
		printf("%lf ", hiddenLayer->_2[k] * 100 );
	}
	printf("\n");
}

Gradient getGradient(W* w, B* b, Data *data, HiddenLayer *hiddenLayer, double h ){
	Gradient gradient;
	gradient.w = (double*)malloc( sizeof(W) );
	gradient.b = (double*)malloc( sizeof(B) );
	
	double loss = getLoss( w, b, data, hiddenLayer );
	double* p = (double*)w;
	for(int i=0; i< sizeof(W)/sizeof(double); i++){
		*(p+i) += h;
		gradient.w[i] = ( getLoss(w, b, data, hiddenLayer ) - loss )/ h;
		*(p+i) -= h;
	}
	p = (double*)b;
	for(int i=0; i< sizeof(B)/sizeof(double); i++){
		*(p+i) += h;
		gradient.b[i] = ( getLoss(w, b, data, hiddenLayer ) - loss )/ h;
		*(p+i) -= h;
	}
	return gradient;
}


void forward(W* w, B* b, Data *data, HiddenLayer *hiddenLayer){
	multiplyMatrices( w->_0, data->input, node_0, batch, 1, hiddenLayer->_0);
	addMatrices( b->_0, hiddenLayer->_0, hiddenLayer->_0, node_0, 1 );
//	sigmoid( hiddenLayer->_0, node_0 );

	multiplyMatrices( w->_1, hiddenLayer->_0, node_1, node_0, 1, hiddenLayer->_1 );	
	addMatrices( b->_1, hiddenLayer->_1, hiddenLayer->_1, node_1, 1 );
//	sigmoid( hiddenLayer->_1, node_1 );

	multiplyMatrices( w->_2, hiddenLayer->_1, node_2, node_1, 1, hiddenLayer->_2 );	
	addMatrices( b->_2, hiddenLayer->_2, hiddenLayer->_2, node_2, 1 );
//	sigmoid( hiddenLayer->_2, node_2 );

	fixedSoftmax( hiddenLayer->_2, outputNode, base );
//	originalSoftmax( hiddenLayer->_2, 2 );
}

double GetDistance(double *realValue, double *predictedValue){
	double result[outputNode];
	subtractMatrices( realValue, predictedValue, result, outputNode, 1);
	return getNorm( result, outputNode );
}

double getLoss(W* w, B* b, Data *data, HiddenLayer *hiddenLayer){
	forward(w, b, data, hiddenLayer );
	//return GetDistance( &(data->output), &(hiddenLayer->_2) );	
	return (-1) * GetDistance( &(data->output), &(hiddenLayer->_2) );	
}

double getRandom(){
	int r = rand();
	if( r % 2 ==0 ){
		r *= -1;
		return (double)r/(double)RAND_MAX;
	}else{
		return (double)r/(double)RAND_MAX;
	}
}

//need to be fixed 
void initParameter( double* w, unsigned int row, unsigned int column){
	for( int i=0; i < row * column; i++){
		w[i]= getRandom();
		//w[i] = 1;
	}
}
