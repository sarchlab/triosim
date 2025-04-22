// Package timemodel provides a performance model for the time of execution of
// operators and layers.
package timemodel

// A TimeEstimatorInput represents the input of a time estimator.
type TimeEstimatorInput struct {
	Name              string
	InputSize         []int
	OutputSize        []int
	RecordedTimeInSec float64
	GPUID             int
}

// A TimeEstimatorOutput represents the output of a time estimator.
type TimeEstimatorOutput struct {
	// The estimated execution time in seconds.
	TimeInSec float64
}

// TimeEstimator estimates the execution time of an operator or a layer.
type TimeEstimator interface {
	// Estimate estimates the execution time of an operator or a layer.
	Estimate(input TimeEstimatorInput) (TimeEstimatorOutput, error)
}

// A AlwaysOneTimeEstimator always returns 1 as the estimated execution time.
type AlwaysOneTimeEstimator struct{}

// Estimate always returns 1 as the estimated execution time.
func (e *AlwaysOneTimeEstimator) Estimate(
	input TimeEstimatorInput,
) (TimeEstimatorOutput, error) {
	return TimeEstimatorOutput{
		TimeInSec: 1,
	}, nil
}

// A RecordedTimeEstimator estimates the execution time of an operator or a
// layer based on the recorded time.
type RecordedTimeEstimator struct{}

// Estimate estimates the execution time of an operator or a layer based on the
// recorded time.
func (e *RecordedTimeEstimator) Estimate(
	input TimeEstimatorInput,
) (TimeEstimatorOutput, error) {
	return TimeEstimatorOutput{
		TimeInSec: input.RecordedTimeInSec,
	}, nil
}
