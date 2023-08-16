#include "header.h"

Data testData[ testBatch ];
double mnist[ testDataSize ][28*28+1];
	
int main()
{
	srand(time(NULL));

	readMnist( testDataPath , mnist, testDataSize );

	for(int i=0;i< testBatch ;i++){
		int random = rand() % testDataSize;
		memcpy( testData[i].input, &mnist[ random ][1], sizeof(double)*inputNode );   	
		normalize( testData[i].input, inputNode, 255 );
		oneHotEncoding( mnist[ random ][0], testData[i].output , outputNode);
	}

	HiddenLayer hiddenLayer;
	W w;
	B b;

	readParameter( parameterFilePath, &w, &b );

	int numberOfCorrect=0;
	printf("prediction start\n\n");
	for(int i=0;i< testBatch ; i++){
		predict( &w, &b, testData+i, &hiddenLayer, &numberOfCorrect );
		printf("\n");
	}
	double accuracy = (double)numberOfCorrect / (double)testBatch ;
	accuracy *= 100;
	printf("accuracy : %lf\n", accuracy );

	return 0;
}

