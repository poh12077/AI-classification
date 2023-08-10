/*
double getLoss(double *realValue, double *predictedValue);
void forward(W* w, B* b, Data* data, HiddenLayer* hiddenLayer);
void initParameter( double* w, unsigned int row, unsigned int column);
double getRandom();
*/
void originalSoftmax(double* vector, int size);
void sigmoid(double* vector, int size);
void fixedSoftmax(double* vector, int size, double base);
double absolute(double x);
double getSqrt( double x );
double getNorm( double *v, unsigned size );
void multiplyMatrices(double *a_, double *b_, unsigned int a_row, unsigned int a_column, unsigned int b_column, double *result_  );
void addMatrices(double *a, double *b, double *result, unsigned int row, unsigned int column );
void subtractMatrices(double *a, double *b, double *result, unsigned int row, unsigned int column );
int findBiggestOrder(double* arr, int size);
