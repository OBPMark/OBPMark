#include <stdlib.h>
#include <math.h>
#include "Preprocessor.h"
#include "Config.h"

#define min(x, y) (((x) < (y)) ? (x) : (y))

/* Minimum maximum definition for the PredictorErrorMapper */
#if SIGNED // fixmevori: Is Signed really required?
int x_min = pow(-2,n_bits-1);
int x_max = pow( 2,n_bits-1) - 1;
#else
int x_min = 0;
int x_max = pow( 2,n_bits) - 1;
#endif

int DelayedStack = 0;

int Preprocessor(int x) 
{
    const int PredictedValue = UnitDelayPredictor(x);
    const int PredictionError = x - PredictedValue;
    const int PreprocessedSample = PredictorErrorMapper(PredictedValue, PredictionError);
    return PreprocessedSample;
}


int UnitDelayPredictor(int DataSample) 
{
    const int CachedDelayedStack = DelayedStack;
    DelayedStack = DataSample;
    return CachedDelayedStack;
}


int PredictorErrorMapper(int PredictedValue, int PredictionError)
{
    const int theta = min(PredictedValue - x_min, x_max-PredictedValue);
    int PreprocessedSample = theta + abs(PredictionError);

    if(0 <= PredictionError && PredictionError <= theta)
    {
        PreprocessedSample = 2 * PredictionError;
    }
    else if(-theta <= PredictionError && PredictionError < 0)
    {
        PreprocessedSample = (2 * abs(PredictionError)) - 1;
    }

    return PreprocessedSample;
}
