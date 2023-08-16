#include "header.h"

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


