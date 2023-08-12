#include "header.h"
#include <math.h>



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
//  sigmoid( hiddenLayer->_0, node_0 );

    multiplyMatrices( w->_1, hiddenLayer->_0, node_1, node_0, 1, hiddenLayer->_1 );
    addMatrices( b->_1, hiddenLayer->_1, hiddenLayer->_1, node_1, 1 );
//  sigmoid( hiddenLayer->_1, node_1 );

    multiplyMatrices( w->_2, hiddenLayer->_1, outputNode, node_1, 1, hiddenLayer->_2 );
    addMatrices( b->_2, hiddenLayer->_2, hiddenLayer->_2, outputNode, 1 );
//  sigmoid( hiddenLayer->_2, outputNode );

    fixedSoftmax( hiddenLayer->_2, outputNode );
//  originalSoftmax( hiddenLayer->_2, 2 );
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



void fixedSoftmax(double* vector, int size ){
	double sum=0;
	for(int i=0;i<size;i++){
		sum += pow(softmaxBase, vector[i]);	
	}
	if(sum ==0 ){
		printf("[error] fixedSoftware\n");  
		exit(-1);
	}

	for(int i=0;i<size;i++){
		vector[i] = pow(softmaxBase, vector[i]) / sum;
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
