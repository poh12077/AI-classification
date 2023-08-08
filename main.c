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
#define learningRateMacro 100
#define base 1.001
#define iteration 10
#define dataNum 2

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
void predict( W* w, B* b, Data *data, HiddenLayer *hiddenLayer );
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

pthread_mutex_t key = PTHREAD_MUTEX_INITIALIZER;

int main()
{

	srand(time(NULL));
	Data data[dataNum];

	//unsigned char mnist[100][28*28+1];
	double mnist[dataNum][28*28+1];
	
	readMnist( "../data/MNIST_CSV/mnist_test.csv", mnist, dataNum);
/*
	memcpy( data[0].input, &mnist[0][1], sizeof(double)*batch );   	
	normalize( data[0].input, batch, 255 );
	oneHotEncoding( mnist[0][0], data[0].output , outputNode);
*/
	for(int i=0;i<dataNum;i++){
		memcpy( data[i].input, &mnist[i][1], sizeof(double)*batch );   	
		normalize( data[i].input, batch, 255 );
		oneHotEncoding( mnist[i][0], data[i].output , outputNode);
	}


//	readImage( "./data/lena.raw", &data[0], 1 );
//	readImage( "./data/barbara.raw", &data[1], 0 );
/*
	//generate random image
	for(int j=1; j<dataNum; j++){
		for(int i=0;i<batch;i++){
			int r = rand()%256;
			data[j].input[i] = (double)r;
		}
		data[j].output[0] = 0;
		data[j].output[1] = 1;
		normalize( data[j].input, batch, 255 );
	}
*/
	HiddenLayer hiddenLayer;
	W w;
	B b;
	initParameter( w._0, node_0, batch);
	initParameter( w._1, node_1, node_0);
	initParameter( w._2, node_2, node_1);
	initParameter( b._0, node_0, 1);
	initParameter( b._1, node_1, 1);
	initParameter( b._2, node_2, 1);

	gradientDescent(&w, &b, data, &hiddenLayer, learningRateMacro );	
/*	
	GradientDescentInput gradientDescentInput[dataNum];
	for(int i=0;i< dataNum ;i++){
		gradientDescentInput[i].w = &w;
		gradientDescentInput[i].b = &b;
		gradientDescentInput[i].data = &data[i];
		gradientDescentInput[i].hiddenLayer = &hiddenLayer;
		gradientDescentInput[i].learningRate = learningRateMacro;
	}

	pthread_t thread[ dataNum ];
	for(int i=0;i< dataNum ;i++){
		pthread_create( &thread[i], NULL, runTraining, (void*)&gradientDescentInput[i] ); 
	}

	for(int i=0;i< dataNum ;i++){
		pthread_join( thread[i], NULL );
	}
*/
/*
	runRandomDataTest(10, &w, &b, &hiddenLayer);
	printf("lena prediction : ");
	predict( &w, &b, &data[0], &hiddenLayer );
	printf("barbara prediction : ");
	predict( &w, &b, &data[1], &hiddenLayer );
*/

	printf("prediction : ");
	predict( &w, &b, &data[1], &hiddenLayer );
	
	return 0;
}

void gradientDescent(W* w, B* b, Data* data, HiddenLayer* hiddenLayer, double learningRate ){	
	static int count=0;
	Gradient gradient;
	double loss_1 = absolute( getLoss( w, b, data, hiddenLayer ) );
	double loss_2;
	double* pw = (double*)w;
	double* pb = (double*)b;
	//while(1)
	for(int c=0; c<iteration; c++) 
	{
		gradient = getGradient(w, b, data, hiddenLayer, 0.001);
		for(int i=0; i< sizeof(W)/sizeof(double); i++){
			pw[i] += learningRate * gradient.w[i]; 
		}
		for(int i=0; i< sizeof(B)/sizeof(double); i++){
			pb[i] += learningRate * gradient.b[i]; 
		}
		free(gradient.w);
		free(gradient.b);
		loss_2 = absolute( getLoss( w, b, data, hiddenLayer ) );
	
		if( loss_1 < loss_2 ){				
			printf("%lf\n", loss_2);
			for(int k=0;k<outputNode;k++){
				printf("%lf ", hiddenLayer->_2[k] );
			}
			printf("\n");
			return;
		}else{
			loss_1 = loss_2;
			printf("%lf\n", loss_2);
			for(int k=0;k<outputNode;k++){
				printf("%lf ", hiddenLayer->_2[k] );
			}
			printf("\n");
		}
	}
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
	//for( unsigned int i=0; i< ; i++ ){
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


void readImage(const char* path, Data* data, int isLena ){
	FILE *fptr;
	if ((fptr = fopen( path ,"rb")) == NULL){
		printf("Error! opening file");
		exit(1);
	}
	unsigned char buffer[resolution];
	fread(buffer, 1 , resolution, fptr); 

	for(int i=0;i<batch;i++){
		int n = resolution/batch;
		data->input[i] = (double)buffer[i*n];
	}
	if( isLena ){
		data->output[0] = 1;
		data->output[1] = 0;
	}else{
		data->output[0] = 0;
		data->output[1] = 1;
	}
	normalize( data->input, batch, 255 );
	
	fclose(fptr); 
}

void normalize( double* data, int size, double max ){
	for( int i=0;i<size;i++){
		data[i] = data[i]/max;
	}
}

void* runTraining(void *input){
	GradientDescentInput* p = (GradientDescentInput*)input;
	for(int i=0;i<iteration;i++){
		pthread_mutex_lock(&key);
		pid_t processId = getpid();
		pthread_t threadId = pthread_self();
		printf(" thread id : %d, process id : %d\n", pthread_self() , processId );
		gradientDescent(p->w, p->b, p->data, p->hiddenLayer, p->learningRate );
		pthread_mutex_unlock(&key);
		usleep(1);
	}
}

void runRandomDataTest(int trials, W* w, B* b, HiddenLayer *hiddenLayer ){
	printf("test started\n");
	Data data;
	for(int j=0; j<trials; j++){
		for(int i=0;i<batch;i++){
			int r = rand()%256;
			data.input[i] = (double)r;
		}
		normalize( data.input, batch, 255 );
		predict( w, b, &data, hiddenLayer );
	}
}

void predict( W* w, B* b, Data *data, HiddenLayer *hiddenLayer ){
	forward(w, b, data, hiddenLayer );
	for(int k=0;k<outputNode;k++){
		printf("%lf ", hiddenLayer->_2[k] );
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

