// Package triosim provides a simulator that replays DNN execution traces.
package triosim

import (
	"encoding/csv"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
)

// A TensorType represent the type of data it stores
type TensorType int

// TensorType constants
const (
	Input TensorType = iota
	Output
	Weight
	RunningMean
	RunningVar
	Bias
	Activation
	Gradient
	Other
)

// TensorMemoryStatus represents the memory status of a tensor.
type TensorMemoryStatus int

// TensorMemoryStatus constants
const (
	TensorMemoryStatusUnknown TensorMemoryStatus = iota
	TensorMemoryStatusAllocated
	TensorMemoryStatusAvailable
	TensorMemoryStatusToBeUsed
	TensorMemoryStatusUsed
)

// A Tensor represents a tensor being used in the neural network. We do not
// carry the data since the execution time should be data independent.
type Tensor struct {
	Index        int
	ID           string //tensor id
	Size         int
	Category     TensorType
	ChunkID      int
	GPUID        int
	MemoryStatus TensorMemoryStatus
}

// Bytes returns the number of bytes of the tensor.
func (t *Tensor) Bytes() uint64 {
	return uint64(t.Size)
}

// A Layer represents a layer in the neural network.
type Layer struct {
	ID           int //operator id
	Name         string
	Inputs       []Tensor
	Outputs      []Tensor
	InputSize    []int
	OutputSize   []int
	TimeInSec    float64
	GPUID        int
	Stage        string
	SetBatchSize bool
	TPflag       int
}

// Trace represents a trace of the execution of the neural network.
type Trace []*Layer

// A TraceLoader loads a trace from a set of files.
type TraceLoader struct {
	// The directory where the trace files are located.
	Dir string
}

// Load loads a trace from a set of files.
func (l *TraceLoader) Load(bsRatio float64) (Trace, error) {
	tensors, err := l.readTensors()
	if err != nil {
		return nil, err
	}

	for key, tensor := range tensors {
		tensor.Size = int(float64(tensor.Size) / bsRatio)
		tensors[key] = tensor
	}

	layers, err := l.readLayers(tensors)
	if err != nil {
		return nil, err
	}
	for _, layer := range layers {
		for i, input := range layer.InputSize {
			layer.InputSize[i] = int(float64(input) / bsRatio)
		}
		for i, output := range layer.OutputSize {
			layer.OutputSize[i] = int(float64(output) / bsRatio)
		}
		layer.TimeInSec = layer.TimeInSec / bsRatio
	}
	return layers, nil
}

// readTensors reads a list of tensors from a CSV file.
func (l *TraceLoader) readTensors() (map[string]Tensor, error) {
	path := filepath.Join(l.Dir, "tensor.csv")
	// path := filepath.Join(l.Dir, "tensor-eviction.csv")
	absPath, err := filepath.Abs(path)
	if err != nil {
		return nil, err
	}

	f, err := os.Open(absPath)
	if err != nil {
		return nil, err
	}
	defer func() {
		closeErr := f.Close()
		if closeErr != nil {
			panic(closeErr)
		}
	}()

	reader := csv.NewReader(f)
	reader.Comma = ','
	reader.TrimLeadingSpace = true

	records, err := reader.ReadAll()
	if err != nil {
		return nil, err
	}

	tensorMap := make(map[string]Tensor)

	for i, record := range records {
		if i == 0 {
			continue
		}

		tensor, err := l.parseTensor(record)
		if err != nil {
			return nil, err
		}
		tensorMap[tensor.ID] = tensor
	}

	return tensorMap, nil
}

func (l *TraceLoader) parseTensor(record []string) (Tensor, error) {
	index, err := strconv.Atoi(record[0])
	tensorID := record[1]

	if err != nil {
		return Tensor{}, err
	}

	gpuID, err := strconv.Atoi(record[7])
	if err != nil {
		return Tensor{}, err
	}

	size, err := strconv.Atoi(record[3])
	if err != nil {
		return Tensor{}, err
	}
	sizeofeach, err := strconv.Atoi(record[4])
	if err != nil {
		return Tensor{}, err
	}
	size = size * sizeofeach
	category := record[5]
	ca := getCategoryType(category)
	return Tensor{
		ID:           tensorID,
		Size:         size,
		Index:        index,
		GPUID:        gpuID,
		Category:     ca,
		MemoryStatus: TensorMemoryStatusUnknown,
	}, nil
}

// readLayers reads a list of layers from a CSV file.
func (l *TraceLoader) readLayers(tensors map[string]Tensor) (Trace, error) {
	path := filepath.Join(l.Dir, "trace.csv")
	absPath, err := filepath.Abs(path)
	if err != nil {
		return nil, err
	}

	f, err := os.Open(absPath)
	if err != nil {
		return nil, err
	}

	reader := csv.NewReader(f)
	reader.Comma = ','
	reader.TrimLeadingSpace = true

	records, err := reader.ReadAll()
	if err != nil {
		return nil, err
	}

	layers := make(Trace, 0, len(records))

	for i, record := range records {
		if i == 0 {
			continue
		}

		layer, err := l.parseLayer(record, tensors)
		if err != nil {
			return nil, err
		}
		layers = append(layers, layer)
	}

	return layers, nil
}

func (l *TraceLoader) parseLayer(
	record []string,
	tensors map[string]Tensor,
) (*Layer, error) {
	var err error
	layer := &Layer{}

	err = l.parseLayerInfo(record, layer)
	if err != nil {
		return nil, err
	}

	err = l.parseLayerInputOutput(record, layer, tensors)
	if err != nil {
		return nil, err
	}

	err = l.parseLayerExecutionInfo(record, layer)
	if err != nil {
		return nil, err
	}

	return layer, nil
}

func (l *TraceLoader) parseLayerInfo(
	record []string,
	layer *Layer,
) error {
	var err error

	layer.ID, err = strconv.Atoi(record[0])
	if err != nil {
		return err
	}

	layer.Name = record[1]
	layer.Stage = record[9]
	layer.TPflag, err = strconv.Atoi(record[10])
	if err != nil {
		return err
	}
	layer.SetBatchSize = false
	return nil
}

func (l *TraceLoader) parseLayerInputOutput(
	record []string,
	layer *Layer,
	tensors map[string]Tensor,
) error {
	var err error

	layer.Inputs, err = parseTensorList(record[2], tensors)
	if err != nil {
		return err
	}

	layer.Outputs, err = parseTensorList(record[3], tensors)
	if err != nil {
		return err
	}

	layer.InputSize, err = parseTensorSizeList(record[6])
	if err != nil {
		return err
	}
	layer.OutputSize, err = parseTensorSizeList(record[7])
	if err != nil {
		return err
	}

	return nil
}

func (l *TraceLoader) parseLayerExecutionInfo(
	record []string,
	layer *Layer,
) error {
	var err error
	layer.TimeInSec, err = strconv.ParseFloat(record[5], 64) //record[4] or [5]
	if err != nil {
		return err
	}
	layer.TimeInSec = layer.TimeInSec / 1e6

	layer.GPUID, err = strconv.Atoi(record[8])
	if err != nil {
		return err
	}

	return nil
}

func parseTensorList(
	str string,
	tensors map[string]Tensor,
) ([]Tensor, error) {
	delimiter := ";"

	str = strings.Trim(str, "[]")
	str = strings.ReplaceAll(str, " ", "")
	tokens := strings.Split(str, delimiter)

	if len(tokens) == 1 && tokens[0] == "" {
		return nil, nil
	}

	tensorList := make([]Tensor, len(tokens))

	for i, token := range tokens {
		token = strings.Trim(token, "'")
		tensor, ok := tensors[token]
		if !ok {
			return nil, fmt.Errorf("tensor %s not found", token)
		}

		tensorList[i] = tensor
	}

	return tensorList, nil
}

func parseTensorSizeList(
	str string,
) ([]int, error) {
	delimiter := ";"

	str = strings.Trim(str, "[]")
	str = strings.ReplaceAll(str, " ", "")
	tokens := strings.Split(str, delimiter)

	if len(tokens) == 1 && tokens[0] == "" {
		return nil, nil
	}

	tensorSizeList := make([]int, len(tokens))

	for i, token := range tokens {
		token = strings.Trim(token, "'")
		sizeitem, err := strconv.Atoi(token)
		if err != nil {
			return nil, err
		}
		tensorSizeList[i] = sizeitem
	}

	return tensorSizeList, nil
}

func getCategoryType(category string) TensorType {
	categoryMap := map[string]TensorType{
		"input":        Input,
		"output":       Output,
		"weight":       Weight,
		"running_mean": RunningMean,
		"running_var":  RunningVar,
		"activation":   Activation,
		"grad_output":  Gradient,
		"grad":         Gradient, //new added from this line
		"bias":         Bias,
		"mat2":         Other,
		"mean":         RunningMean,
		"rstd":         Other,
		"total_weight": Weight,
	}
	if t, ok := categoryMap[category]; ok {
		return t
	}
	return Other
}
