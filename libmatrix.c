#include "header.h"
#include <math.h>

int findBiggestOrder(double* arr, int size){
	double* brr = (double*)malloc( sizeof(double) * size );
	memcpy( brr, arr, sizeof(double) * size ); 
	int orderOfBiggest=0;
	for(int i=1;i<size;i++){
		if( brr[0] < brr[i] ){
			brr[0] = brr[i];
			orderOfBiggest = i;
		}
	}
	free(brr);
	return orderOfBiggest;
}

void multiplyMatrices(double *a_, double *b_, unsigned int a_row, unsigned int a_column, unsigned int b_column, double *result_  ){
	double (*a)[a_column] = a_;
	double (*b)[b_column] = b_;
	double (*result)[b_column] = result_;
	
	unsigned int i,j,k;
	double sum=0;
	for(i=0; i < a_row; i++){
		for(k=0; k < b_column; k++){
			for(j=0; j < a_column; j++){
				sum += a[i][j] * b[j][k];
			}
			result[i][k] = sum;
			sum=0;
		}
	}	
}

void sigmoid(double* vector, int size){
	for(int i=0;i<size;i++){
		vector[i] = exp(vector[i]) / ( exp(vector[i]) + 1 ) ;	
	}
	for(int i=0;i<size;i++){
		if(	isinf( vector[i] ) || isnan( vector[i] ) ){
			printf("[error] sigmoid\n");  
			exit(-1);
		}
	}
}

void originalSoftmax(double* vector, int size){
/*
	if ( vector[0] < vector[1] ){
		vector[1] = vector[1] - vector[0];
		vector[0] = 0;
	}else{
		vector[0] = vector[0] - vector[1];
		vector[1] = 0;
	}
*/
	double sum=0;
	for(int i=0;i<size;i++){
		sum += exp( vector[i] );	
	}
	if(sum ==0 ){
		printf("[error] fixedSoftware\n");  
		exit(-1);
	}

	for(int i=0;i<size;i++){
		vector[i] = exp( vector[i] ) / sum;	
	}
	for(int i=0;i<size;i++){
		if(	isinf( vector[i] ) || isnan( vector[i] ) ){
			printf("[error] fixedSoftware\n");  
			exit(-1);
		}
	}
}

void fixedSoftmax(double* vector, int size, double base){
	double sum=0;
	for(int i=0;i<size;i++){
		sum += pow(base, vector[i]);	
	}
	if(sum ==0 ){
		printf("[error] fixedSoftware\n");  
		exit(-1);
	}

	for(int i=0;i<size;i++){
		vector[i] = pow(base, vector[i]) / sum;
	}
	for(int i=0;i<size;i++){
		if(	isinf( vector[i] ) || isnan( vector[i] ) ){
			printf("[error] fixedSoftware\n");  
			exit(-1);
		}
	}
}

double absolute( double x){
	if( x<0){
		return (-1)*x;
	}else{
		return x;
	}
}

double getNorm( double *v, unsigned size ){
	double sum=0;
	for( unsigned int i=0; i < size; i++){
		sum += v[i]*v[i];
	}
	return sqrt(sum);
}

double getSqrt( double x ){
	return x;
}

void addMatrices(double *a, double *b, double *result, unsigned int row, unsigned int column ){
	for(unsigned int i=0; i < row * column; i++){
		result[i] = a[i] + b[i];
	}	
}

void subtractMatrices(double *a, double *b, double *result, unsigned int row, unsigned int column ){
	for(unsigned int i=0; i < row * column; i++){
		result[i] = a[i] - b[i];
	}	
}

/*
int main(){
	
	double a[5]={ -1.3, 1, -2.4, 5.4, 8.2};
	int x = findBiggestOrder(a,5);

}
*/
