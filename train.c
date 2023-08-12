
#include "header.h"

/*
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdlib.h>





#define inputNode 784
#define node_0 10
#define node_1 10
#define outputNode 10
#define learningRateMacro 0.1 //traing work 
#define base 1.001
#define iteration 1
#define trainBatch 1000
#define parameterFilePath "parameter"
#define trainDataPath "../data/MNIST_CSV/mnist_train.csv"
#define trainDataSize 60000

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
*/


Data trainData[trainBatch];
double mnist[ trainDataSize ][28*28+1];
	
int main()
{

	srand(time(NULL));

	HiddenLayer hiddenLayer;
	W w;
	B b;

	initParameter( w._0, node_0, inputNode);
	initParameter( w._1, node_1, node_0);
	initParameter( w._2, outputNode, node_1);
	initParameter( b._0, node_0, 1);
	initParameter( b._1, node_1, 1);
	initParameter( b._2, outputNode, 1);

	while(1){

		readMnist( trainDataPath , mnist, trainDataSize );

		for(int i=0;i<trainBatch;i++){
			int random = rand() % trainDataSize;
			memcpy( trainData[i].input, &mnist[ random ][1], sizeof(double)*inputNode );   	
			normalize( trainData[i].input, inputNode, 255 );
			oneHotEncoding( mnist[ random ][0], trainData[i].output , outputNode);
		}
		readParameter( parameterFilePath, &w, &b );

		for(int j=0;j<iteration;j++){
			for(int i=0;i<trainBatch;i++){
				printf("data number : %d\n", i);
				gradientDescent(&w, &b, trainData+i, &hiddenLayer, learningRateMacro );	
				printf("\n");
			}
		}
		writeParameter( parameterFilePath, &w, &b );
	}
	return 0;
}

/*

void gradientDescent(W* w, B* b, Data* data, HiddenLayer* hiddenLayer, double learningRate ){	
	static int count=0;
	Gradient gradient;
	double loss_1 = absolute( getLoss( w, b, data, hiddenLayer ) );
	double loss_2;
	double* pw = (double*)w;
	double* pb = (double*)b;
	//while(1)
	//for(int c=0; c<iteration; c++) 
	{
		gradient = getGradient(w, b, data, hiddenLayer, 0.0001);
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
			printf("[ lose : %lf ]\n ", loss_2);
			for(int k=0;k<outputNode;k++){
				printf("%lf ", hiddenLayer->_2[k] );
			}
			printf("\n");
		}
	}
}

void writeParameter(char* path, W* w, B* b ){
	FILE *fptr;
	fptr = fopen(path,"wb");
	if(fptr == NULL)
	{
	  printf("[writeParameter Error]\n");
	}

	fwrite( w, sizeof(W), 1, fptr);
	fwrite( b, sizeof(B), 1, fptr);

	fclose(fptr);
}

void readParameter(char* path, W* w, B* b ){
	FILE *fptr;
	fptr = fopen(path,"rb");
	if(fptr == NULL)
	{
		printf("initialize parameter\n");
		return;
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
	if ((fptr = fopen( path ,"r")) == NULL){
		printf(" [readMnist  Error! ]\n");
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
	multiplyMatrices( w->_0, data->input, node_0, inputNode, 1, hiddenLayer->_0);
	addMatrices( b->_0, hiddenLayer->_0, hiddenLayer->_0, node_0, 1 );
//	sigmoid( hiddenLayer->_0, node_0 );

	multiplyMatrices( w->_1, hiddenLayer->_0, node_1, node_0, 1, hiddenLayer->_1 );	
	addMatrices( b->_1, hiddenLayer->_1, hiddenLayer->_1, node_1, 1 );
//	sigmoid( hiddenLayer->_1, node_1 );

	multiplyMatrices( w->_2, hiddenLayer->_1, outputNode, node_1, 1, hiddenLayer->_2 );	
	addMatrices( b->_2, hiddenLayer->_2, hiddenLayer->_2, outputNode, 1 );
//	sigmoid( hiddenLayer->_2, outputNode );

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

void initParameter( double* w, unsigned int row, unsigned int column){
	for( int i=0; i < row * column; i++){
		w[i]= getRandom();
		//w[i] = 1;
	}
}

*/

