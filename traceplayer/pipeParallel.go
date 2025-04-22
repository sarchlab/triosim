// Package traceplayer provides a trace player that plays a trace and
// simulates the execution of the trace.
package traceplayer

import (
	"fmt"
	"reflect"
	"strconv"

	"github.com/syifan/triosim"
	"github.com/syifan/triosim/timemodel"
	"gitlab.com/akita/akita/v3/sim"
)

// PipeMemoryRegion describes a region of memory that can hold tensors.
type PipeMemoryRegion struct {
	Name            string
	CapacityLimited bool
	Capacity        uint64
	GPUID           int

	Tensors                 map[string]triosim.Tensor
	pipeInflightTransfer    map[int][]*triosim.TensorMsg
	LayersIDPerRegion       []int
	pipeDoingComputing      []bool
	pipeFetchingLayerIndex  []int
	pipeComputingLayerIndex []int
	finalbackwardlayerid    int
	finalforwardLayerID     int
}

// TotalUtilizedBytes returns the total number of bytes used by the tensors in the region.
func (r *PipeMemoryRegion) TotalUtilizedBytes() uint64 {
	var total uint64

	for _, t := range r.Tensors {
		total += t.Bytes()
	}

	return total
}

// A PipeParallelTracePlayer replays the forward path of a trace.
type PipeParallelTracePlayer struct {
	*sim.ComponentBase

	sim.TimeTeller
	sim.EventScheduler
	timeEstimator timemodel.TimeEstimator

	memoryRegions       []*PipeMemoryRegion
	defaultMemoryRegion *PipeMemoryRegion

	trace              triosim.Trace
	batchSize          int
	pipeAction         []string
	microBatchSize     int
	numRound           int
	MaxForwardLayerID  int
	MaxBackwardLayerID int
	stall              bool
	stallInfos         [][]int
	roundstallInfos    [][]int
}

// NewPipeParallelTracePlayer creates a new PipeParallelTracePlayer.
func NewPipeParallelTracePlayer(
	name string,
	tt sim.TimeTeller,
	es sim.EventScheduler,
	timeEstimator timemodel.TimeEstimator,
) *PipeParallelTracePlayer {
	p := &PipeParallelTracePlayer{
		timeEstimator:  timeEstimator,
		TimeTeller:     tt,
		EventScheduler: es,
	}

	p.ComponentBase = sim.NewComponentBase(name)

	return p
}

// AddMemoryRegion adds a memory region to the player.
func (p *PipeParallelTracePlayer) AddMemoryRegion(
	region *PipeMemoryRegion,
	port sim.Port,
) {
	p.memoryRegions = append(p.memoryRegions, region)
	p.AddPort(region.Name, port)
}

// SetDefaultMemoryRegion sets the default memory region.
func (p *PipeParallelTracePlayer) SetDefaultMemoryRegion(region *PipeMemoryRegion) {
	p.defaultMemoryRegion = region
}

// Handle function of a PipeParallelTracePlayer handles events.
func (p *PipeParallelTracePlayer) Handle(e sim.Event) error {
	switch e := e.(type) {
	case playNextPipelineEvent:
		gpuID := e.gpuID
		roundID := e.roundID
		p.playNextPipeline(gpuID, roundID)
	case pipeLayerCompletionEvent:
		p.completePipeLayer(e)
	default:
		panic("PipeParallelTracePlayer cannot handle this event type " +
			reflect.TypeOf(e).String())
	}

	return nil
}

// NotifyPortFree function of a PipeParallelTracePlayer notifies that the
// one port of the component if free.
func (p *PipeParallelTracePlayer) NotifyPortFree(
	now sim.VTimeInSec,
	port sim.Port,
) {
	msg := port.Retrieve(now)
	msginfo := msg.(*triosim.TensorMsg)
	if msginfo.Purpose == "nextGPU" {
		fmt.Println("Send message to next GPU successfully and port free")
	}
	p.playNextPipeline(msginfo.GPUID, msginfo.RoundID)
}

// NotifyRecv function notifies that the component has received a message.
func (p *PipeParallelTracePlayer) NotifyRecv(
	now sim.VTimeInSec,
	port sim.Port,
) {
	msg := port.Retrieve(now)
	var gpuID int
	switch msg := msg.(type) {
	case *triosim.TensorMsg:
		p.recvTensorPkg(msg)
		gpuID = msg.GPUID
		if msg.Purpose == "nextGPU" {
			fmt.Println("Send message to next GPU successfully")
		}
		if msg.Purpose == "nextRound" {
			p.NextRoundPipelineStart()
		}
		p.scheduleNextPipeline(gpuID, msg.RoundID)
	default:
		panic(fmt.Sprintf("Cannot handle message %T", msg))
	}
}

func (p *PipeParallelTracePlayer) recvTensorPkg(msg *triosim.TensorMsg) {
	p.removeInflightTransfer(msg)
	p.addTensorsToMemRegion(msg.TensorPkg, msg.GPUID, msg.Purpose, msg.RoundID)
}

func (p *PipeParallelTracePlayer) removeInflightTransfer(msg *triosim.TensorMsg) {
	removed := false
	gpuID := msg.GPUID
	region := p.memoryRegions[gpuID]
	for i, m := range region.pipeInflightTransfer[msg.RoundID] {
		if m == msg {
			region.pipeInflightTransfer[msg.RoundID] = append(
				region.pipeInflightTransfer[msg.RoundID][:i],
				region.pipeInflightTransfer[msg.RoundID][i+1:]...,
			)
			removed = true
			break
		}
	}
	if !removed {
		panic("Cannot find the message in inflight")
	}
}

// SetTrace sets the trace to replay by the PipeParallelTracePlayer.
func (p *PipeParallelTracePlayer) SetTrace(
	trace triosim.Trace,
	batchSize int,
) {
	if p.defaultMemoryRegion == nil {
		panic("DefaultMemoryRegion is not set")
	}

	p.addTensorsToDefaultMemRegion(trace)

	p.trace = trace
	p.batchSize = batchSize
}

func (p *PipeParallelTracePlayer) addTensorsToDefaultMemRegion(
	trace triosim.Trace,
) {
	for _, layer := range trace {
		for _, tensor := range layer.Inputs {
			p.defaultMemoryRegion.Tensors[tensor.ID] = tensor
		}
	}
}

func (p *PipeParallelTracePlayer) msgPkgToSend(
	srcRegion string,
	dstRegion string,
	tensors []triosim.Tensor,
	gpuID int,
	purpose string,
	roundID int,
) bool {
	src := p.GetPortByName(srcRegion)
	dst := p.GetPortByName(dstRegion)
	totalBytes := 0
	for _, tensor := range tensors {
		totalBytes += int(tensor.Bytes())
	}
	msg := &triosim.TensorMsg{
		TensorPkg:     tensors,
		DstRegionName: dstRegion,
		GPUID:         gpuID,
		RoundID:       roundID,
		Purpose:       purpose,
		MsgMeta: sim.MsgMeta{
			ID:           sim.GetIDGenerator().Generate(),
			Src:          src,
			Dst:          dst,
			SendTime:     p.CurrentTime(),
			TrafficBytes: totalBytes,
		},
	}
	err := src.Send(msg)
	if err == nil {
		region := p.memoryRegions[gpuID]
		region.pipeInflightTransfer[msg.RoundID] = append(
			region.pipeInflightTransfer[msg.RoundID], msg)
	}
	return err == nil
}

func (p *PipeParallelTracePlayer) nextTensorPkgToMove(
	layer *triosim.Layer,
	gpuID int,
) (tensors []triosim.Tensor, exist bool) {
	for _, tensor := range layer.Inputs {
		if tensor.Category != triosim.Gradient && !p.isTensorReady(tensor, gpuID) {
			tensors = append(tensors, tensor)
		}
	}

	if tensors != nil {
		return tensors, true
	}

	return nil, false
}

func (p *PipeParallelTracePlayer) isTensorReady(
	tensor triosim.Tensor,
	gpuID int,
) bool {
	region := p.memoryRegions[gpuID]

	_, ok := region.Tensors[tensor.ID]

	return ok
}

func (p *PipeParallelTracePlayer) allTensorsOfLayerAreAvailable(
	layer *triosim.Layer,
	gpuID int,
) bool {
	region := p.memoryRegions[gpuID]
	requiredTensors := append(layer.Inputs, layer.Inputs...)
	i := 0
	for _, tensor := range requiredTensors {
		_, ok := region.Tensors[tensor.ID]
		if ok {
			i++
		}
	}

	return i == len(requiredTensors)
}

func (p *PipeParallelTracePlayer) checkSpaceForTensors(
	tensor []triosim.Tensor,
	gpuID int,
	roundID int,
) bool {
	totalTensorBytes := uint64(0)
	for _, t := range tensor {
		totalTensorBytes += t.Bytes()
	}

	region := p.memoryRegions[gpuID]
	totalBytes := region.TotalUtilizedBytes() + totalTensorBytes
	if totalBytes <= region.Capacity {
		return true
	}

	p.removeTensorFromMemRegion(gpuID, roundID)

	totalBytes = region.TotalUtilizedBytes() + totalTensorBytes
	if totalBytes <= region.Capacity {
		return true
	}

	fmt.Println("region is full after deleting")
	return false
}

func (p *PipeParallelTracePlayer) addTensorsToMemRegion(
	tensors []triosim.Tensor,
	gpuID int,
	purpose string,
	roundID int,
) bool {
	region := p.memoryRegions[gpuID]
	tensorsCheck := p.filterTensors(tensors, gpuID, purpose, roundID)
	if len(tensorsCheck) == 0 {
		return true
	}
	if p.checkSpaceForTensors(tensorsCheck, gpuID, roundID) {
		var Status triosim.TensorMemoryStatus
		switch purpose {
		case "complete":
			Status = triosim.TensorMemoryStatusUsed
		case "fetch":
			Status = triosim.TensorMemoryStatusToBeUsed
		case "allocate":
			Status = triosim.TensorMemoryStatusAllocated
		default:
			Status = triosim.TensorMemoryStatusAvailable
		}
		for _, tensor := range tensorsCheck {
			tensor.MemoryStatus = Status
			region.Tensors[tensor.ID] = tensor
		}
	} else {
		fmt.Println("region is still full-add")
		return false
	}

	return true
}

func (p *PipeParallelTracePlayer) filterTensors(
	tensors []triosim.Tensor,
	gpuID int,
	purpose string,
	roundID int,
) []triosim.Tensor {
	region := p.memoryRegions[gpuID]
	var tensorsCheck []triosim.Tensor

	switch purpose {
	case "fetch":
		tensorsCheck = p.handleFetch(tensors, region)
	default: //"allocate", "complete"
		tensorsCheck = p.handleAllocateComplete(tensors, region, purpose, roundID)
	}

	return tensorsCheck
}

func (p *PipeParallelTracePlayer) handleFetch(
	tensors []triosim.Tensor,
	region *PipeMemoryRegion,
) []triosim.Tensor {
	tensorsCheck := make([]triosim.Tensor, 0)
	for _, tensor := range tensors {
		dstTensor, found := region.Tensors[tensor.ID]
		if found {
			dstTensor.MemoryStatus = triosim.TensorMemoryStatusToBeUsed
			region.Tensors[tensor.ID] = dstTensor
		} else {
			tensorsCheck = append(tensorsCheck, tensor)
		}
	}

	return tensorsCheck
}

func (p *PipeParallelTracePlayer) handleAllocateComplete(
	tensors []triosim.Tensor,
	region *PipeMemoryRegion,
	purpose string,
	roundID int,
) []triosim.Tensor {
	tensorsCheck := make([]triosim.Tensor, 0)
	MemoryStatus := triosim.TensorMemoryStatusAllocated
	if purpose == "complete" {
		MemoryStatus = triosim.TensorMemoryStatusUsed
		tensors = append(tensors, p.trace[region.pipeComputingLayerIndex[roundID]-1].Inputs...)
	}

	for _, tensor := range tensors {
		dstTensor, found := region.Tensors[tensor.ID]
		if found && dstTensor.MemoryStatus == MemoryStatus {
			continue
		}

		if found {
			delete(region.Tensors, tensor.ID)
		}

		tensorsCheck = append(tensorsCheck, tensor)
	}

	return tensorsCheck
}

func (p *PipeParallelTracePlayer) removeTensorFromMemRegion(
	gpuID int,
	roundID int,
) {
	region := p.memoryRegions[gpuID]
	removed := false
	existingTensorIDs := make(map[string]struct{})
	var layer *triosim.Layer
	var LayerTensors []triosim.Tensor

	if region.pipeComputingLayerIndex[roundID] == len(p.trace) {
		fmt.Println("the region is full when the last layer, do not need to store it in the memory.")
		return
	}

	layer = p.trace[region.pipeComputingLayerIndex[roundID]]
	LayerTensors = append(layer.Outputs, layer.Inputs...)

	for _, layertensor := range LayerTensors {
		existingTensorIDs[layertensor.ID] = struct{}{}
	}

	for _, t := range region.Tensors {
		if _, exists := existingTensorIDs[t.ID]; !exists {
			if t.MemoryStatus == triosim.TensorMemoryStatusUsed {
				delete(region.Tensors, t.ID)
				removed = true
			}
		}
	}

	fmt.Println("removeTensorFromMemRegion", removed)
	if !removed {
		fmt.Println("it will happen when the memory is not large enough to hold one layer tensors, which is unreasonable")
	}
}

// A playNextPipelineEvent triggers the player to continue to play the trace when in Pipeline parallelism process.
type playNextPipelineEvent struct {
	time    sim.VTimeInSec
	handler *PipeParallelTracePlayer
	gpuID   int
	roundID int
}

// Time returns the time of the event.
func (e playNextPipelineEvent) Time() sim.VTimeInSec {
	return e.time
}

// Handler returns the handler of the event.
func (e playNextPipelineEvent) Handler() sim.Handler {
	return e.handler
}

// IsSecondary always returns false.
func (e playNextPipelineEvent) IsSecondary() bool {
	return false
}

// GetGPUID returns the GPUID of the event.
func (e playNextPipelineEvent) GetGPUID() int {
	return e.gpuID
}

// GetRoundID returns the RoundID of the event.
func (e playNextPipelineEvent) GetRoundID() int {
	return e.roundID
}

// A pipeLayerCompletionEvent is triggered when a layer is completed.
type pipeLayerCompletionEvent struct {
	time    sim.VTimeInSec
	handler *PipeParallelTracePlayer
	layer   *triosim.Layer
	gpuID   int
	roundID int
}

// GetGPUID returns the GPUID of the event.
func (e pipeLayerCompletionEvent) GetGPUID() int {
	return e.gpuID
}

// GetRoundID returns the RoundID of the event.
func (e pipeLayerCompletionEvent) GetRoundID() int {
	return e.roundID
}

// Time returns the time of the event.
func (e pipeLayerCompletionEvent) Time() sim.VTimeInSec {
	return e.time
}

// Handler returns the handler of the event.
func (e pipeLayerCompletionEvent) Handler() sim.Handler {
	return e.handler
}

// IsSecondary always returns false.
func (e pipeLayerCompletionEvent) IsSecondary() bool {
	return false
}

func (p *PipeParallelTracePlayer) PipelineStart() {
	if (p.trace == nil) || (len(p.trace) == 0) {
		panic("Trace is not set")
	}
	p.numRound = 0
	gpuLayerMappings := p.assignLayersToRegions()
	for _, item := range p.memoryRegions {
		item.Tensors = make(map[string]triosim.Tensor)
		item.LayersIDPerRegion = gpuLayerMappings[item.Name]
		if item.LayersIDPerRegion != nil {
			item.pipeFetchingLayerIndex[p.numRound] = 0
			item.pipeComputingLayerIndex[p.numRound] = 0
		}
	}

	p.pipeAction[p.numRound] = "doforward"
	p.scheduleNextPipeline(0, p.numRound)
}

func (p *PipeParallelTracePlayer) SetMicroBatch(
	microBatchSize int,
) {
	p.microBatchSize = microBatchSize
	for i := 0; i <= p.batchSize/p.microBatchSize; i++ {
		p.pipeAction = append(p.pipeAction, "")
	}
	for _, item := range p.memoryRegions {
		item.pipeDoingComputing = make([]bool, p.batchSize/p.microBatchSize)
		item.pipeFetchingLayerIndex = make([]int, p.batchSize/p.microBatchSize)
		item.pipeComputingLayerIndex = make([]int, p.batchSize/p.microBatchSize)
		item.pipeInflightTransfer = make(map[int][]*triosim.TensorMsg)
	}
	fmt.Println("numRound is set to ", p.batchSize/p.microBatchSize)
}

func (p *PipeParallelTracePlayer) assignLayersToRegions() map[string][]int {
	// Separate forward and backward layers
	forwardLayers := []int{}
	backwardLayers := []int{}
	maxForwardLayerID := -1
	maxBackwardLayerID := -1
	for _, layer := range p.trace {
		if layer.Stage == "forward" {
			forwardLayers = append(forwardLayers, layer.ID)
			if layer.ID > maxForwardLayerID {
				maxForwardLayerID = layer.ID
			}
		} else {
			backwardLayers = append(backwardLayers, layer.ID)
			if layer.ID > maxBackwardLayerID {
				maxBackwardLayerID = layer.ID
			}
		}
	}
	p.MaxForwardLayerID = maxForwardLayerID
	p.MaxBackwardLayerID = maxBackwardLayerID
	numRegions := len(p.memoryRegions) - 1
	// Distribute forward layers among GPUs
	gpuForwardLayers := make(map[string][]int)
	for i := 0; i < numRegions; i++ {
		start := i * len(forwardLayers) / numRegions
		end := (i + 1) * len(forwardLayers) / numRegions
		gpuName := "GPU" + strconv.Itoa(i)
		gpuForwardLayers[gpuName] = forwardLayers[start:end]
		p.memoryRegions[i].finalforwardLayerID = forwardLayers[end-1]
	}

	// Distribute backward layers among GPUs
	gpuBackwardLayers := make(map[string][]int)
	for i := 0; i < numRegions; i++ {
		start := i * len(backwardLayers) / numRegions
		end := (i + 1) * len(backwardLayers) / numRegions
		gpuName := "GPU" + strconv.Itoa(numRegions-1-i)
		gpuBackwardLayers[gpuName] = backwardLayers[start:end]
		p.memoryRegions[numRegions-1-i].finalbackwardlayerid = backwardLayers[end-1]
	}
	// Combine forward and backward layers
	gpuLayerMappings := make(map[string][]int)
	for gpu, layers := range gpuForwardLayers {
		gpuLayerMappings[gpu] = append(gpuLayerMappings[gpu], layers...)
	}
	for gpu, layers := range gpuBackwardLayers {
		gpuLayerMappings[gpu] = append(gpuLayerMappings[gpu], layers...)
	}

	// Example output to verify the mappings
	for gpu, layers := range gpuLayerMappings {
		fmt.Printf("%s manages layers: %v\n", gpu, layers)
	}
	return gpuLayerMappings
}

// playNextPipeline performs the next action that replays the trace when in pipeline parallesim
func (p *PipeParallelTracePlayer) playNextPipeline(gpuID int, roundID int) {
	p.doPipeFetching(gpuID, roundID)
	p.doPipeComputing(gpuID, roundID)
}

func (p *PipeParallelTracePlayer) layerExists(gpuID int, layerID int) bool {
	region := p.memoryRegions[gpuID]
	for _, id := range region.LayersIDPerRegion {
		if id == layerID {
			return true
		}
	}
	return false
}

func (p *PipeParallelTracePlayer) checkLayerIndexInRegions(gpuID int, currLayerIndex int, roundID int) bool {
	layerID := p.trace[currLayerIndex].ID
	region := p.memoryRegions[gpuID]
	if !p.layerExists(gpuID, layerID) { //if it's not in the region, then do not need to fetch/compute
		return true
	}

	prevlayerID := 0 //if it's the 1st layer, then just fetch/compute it
	if currLayerIndex != 0 {
		prevlayerID = p.trace[currLayerIndex-1].ID
	}

	finalLayerID := region.finalforwardLayerID
	if p.pipeAction[roundID] == "dobackward" {
		finalIndex := len(region.LayersIDPerRegion) - 1
		finalLayerID = region.LayersIDPerRegion[finalIndex]
	}
	//if it's the first layer in the layers on the next GPU, the previous layer should be the last layer on the current GPU
	if prevlayerID == finalLayerID &&
		layerID != finalLayerID {
		return true
	}
	return false
}

func (p *PipeParallelTracePlayer) divideMiniToMicroBatch(
	layer *triosim.Layer,
	roundID int,
) {
	if !layer.SetBatchSize {
		numMicroBatch := p.batchSize / p.microBatchSize
		for i := range layer.Inputs {
			layer.Inputs[i].Size = layer.Inputs[i].Size / numMicroBatch
			layer.InputSize[i] = layer.InputSize[i] / numMicroBatch
			layer.Inputs[i].ID = layer.Inputs[i].ID + "Round" + strconv.Itoa(roundID)
		}
		for i := range layer.Outputs {
			layer.Outputs[i].Size = layer.Outputs[i].Size / numMicroBatch
			layer.OutputSize[i] = layer.OutputSize[i] / numMicroBatch
			layer.Outputs[i].ID = layer.Outputs[i].ID + "Round" + strconv.Itoa(roundID)
		}

		layer.TimeInSec = layer.TimeInSec / float64(numMicroBatch)
		layer.SetBatchSize = true
	}
}

func (p *PipeParallelTracePlayer) doPipeFetching(gpuID int, roundID int) {
	region := p.memoryRegions[gpuID]
	if len(region.pipeInflightTransfer[roundID]) > 0 {
		return
	}

	fetchingIndex := region.pipeFetchingLayerIndex[roundID]
	if fetchingIndex >= len(p.trace) {
		return
	}

	if p.checkLayerIndexInRegions(gpuID, fetchingIndex, roundID) {
		return
	}

	if fetchingIndex < region.pipeComputingLayerIndex[roundID] {
		region.pipeFetchingLayerIndex[roundID]++
		p.scheduleNextPipeline(gpuID, roundID)
		return
	}

	p.logStepDone(gpuID, roundID, "fetching check", p.memoryRegions[gpuID].pipeFetchingLayerIndex[roundID])
	if p.allTensorsOfLayerAreAvailable(p.trace[fetchingIndex], gpuID) {
		region.pipeFetchingLayerIndex[roundID]++
		p.doPipeFetching(gpuID, roundID)
		return
	}

	p.divideMiniToMicroBatch(
		p.trace[fetchingIndex], roundID)
	tensors, needsMoving := p.nextTensorPkgToMove(
		p.trace[fetchingIndex], gpuID)
	if !p.addTensorsToMemRegion(tensors, gpuID, "allocate", roundID) {
		return
	}

	p.addTensorsToMemRegion(p.trace[fetchingIndex].Outputs,
		gpuID, "allocate", roundID)
	if !needsMoving {
		region.pipeFetchingLayerIndex[roundID]++
		p.scheduleNextPipeline(gpuID, roundID)
		return
	}

	p.logStepDone(gpuID, roundID, "fetching start", p.memoryRegions[gpuID].pipeFetchingLayerIndex[roundID])
	err := p.msgPkgToSend(p.defaultMemoryRegion.Name, region.Name, tensors, gpuID, "fetch", roundID)
	if !err {
		return
	}
}

func (p *PipeParallelTracePlayer) doPipeComputing(gpuID int, roundID int) {
	region := p.memoryRegions[gpuID]
	if region.pipeDoingComputing[roundID] {
		return
	}
	computeIndex := region.pipeComputingLayerIndex[roundID]
	if computeIndex >= len(p.trace) {
		return
	}

	if p.checkLayerIndexInRegions(gpuID, computeIndex, roundID) {
		return
	}

	layer := p.trace[computeIndex]
	p.logStepDone(gpuID, roundID, "computing check", p.memoryRegions[gpuID].pipeComputingLayerIndex[roundID])
	p.divideMiniToMicroBatch(
		p.trace[computeIndex], roundID)

	_, needsMoving := p.nextTensorPkgToMove(layer, gpuID)
	if needsMoving {
		return
	}

	p.logStepDone(gpuID, roundID, "computing start", p.memoryRegions[gpuID].pipeComputingLayerIndex[roundID])
	inputSize := layer.InputSize
	outputSize := layer.OutputSize

	input := timemodel.TimeEstimatorInput{
		Name:              layer.Name,
		InputSize:         inputSize,
		OutputSize:        outputSize,
		RecordedTimeInSec: layer.TimeInSec,
		GPUID:             gpuID,
	}
	output, err := p.timeEstimator.Estimate(input)
	if err != nil {
		panic(err)
	}

	now := p.CurrentTime()
	evt := pipeLayerCompletionEvent{
		time:    now + sim.VTimeInSec(output.TimeInSec),
		handler: p,
		layer:   layer,
		gpuID:   gpuID,
		roundID: roundID,
	}
	p.Schedule(evt)

	region.pipeDoingComputing[roundID] = true
	region.pipeComputingLayerIndex[roundID]++
}

func (p *PipeParallelTracePlayer) completePipeLayer(e sim.Event) {
	evt := e.(pipeLayerCompletionEvent)
	layer := evt.layer
	gpuID := evt.gpuID
	roundID := evt.roundID
	nextGPU := gpuID
	region := p.memoryRegions[gpuID]
	gpuCount := len(p.memoryRegions) - 1
	p.addTensorsToMemRegion(layer.Outputs, gpuID, "complete", roundID)
	region.pipeDoingComputing[roundID] = false
	p.logStepDone(gpuID, roundID, "computing done", p.memoryRegions[gpuID].pipeComputingLayerIndex[roundID]-1)
	nextlayerID := layer.ID + 1
	computeIndex := region.pipeComputingLayerIndex[roundID]
	if computeIndex < len(p.trace) {
		nextlayerID = p.trace[computeIndex].ID
	}
	fmt.Println("------------action, ", p.pipeAction[roundID])
	finalLayerID := region.finalforwardLayerID
	MaxLayerID := p.MaxForwardLayerID
	if p.pipeAction[roundID] == "dobackward" {
		finalLayerID = region.finalbackwardlayerid
	}

	if layer.ID == finalLayerID &&
		nextlayerID != finalLayerID {
		if p.pipeAction[roundID] == "doforward" {
			if layer.ID == MaxLayerID {
				p.pipeAction[roundID] = "dobackward"
				// The last GPU is currently processing in reverse. Initiate reverse processing for the second-to-last GPU.
				nextGPU = gpuID
				if roundID != p.batchSize/p.microBatchSize-1 {
					p.stall = true
					p.keepStallInfo(gpuID, nextGPU, roundID, layer)
					p.handlePossiblePrevStall(gpuID, roundID)
					return
				}
			} else {
				nextGPU = (gpuID + 1) % gpuCount
			}
		} else {
			nextGPU = (gpuID - 1) % gpuCount
			if nextGPU == -1 {
				p.handleNextStep(gpuID, nextGPU, roundID, layer)
				return //backward is done
			}
			if roundID == p.batchSize/p.microBatchSize-1 {
				p.stall = false
			}
		}
		p.updateMaxFetchComputeIndex(gpuID, nextGPU, roundID)
	}

	fmt.Println("------------ gpu ------------------- " + strconv.Itoa(nextGPU))
	p.handleNextStep(gpuID, nextGPU, roundID, layer)
}

func (p *PipeParallelTracePlayer) handlePossiblePrevStall(gpuID int, roundID int) {
	if len(p.roundstallInfos) > 0 {
		layerIndex := p.memoryRegions[gpuID].pipeComputingLayerIndex[roundID] - 1
		layerID := p.trace[layerIndex].ID
		if p.pipeAction[roundID] == "dobackward" {
			nextRoundID := roundID + 1
			if nextRoundID < p.batchSize/p.microBatchSize {
				if layerID == p.memoryRegions[gpuID].finalforwardLayerID {
					p.removeRoundStallInfo(gpuID, nextRoundID)
				}
			}
		}
	}
}

func (p *PipeParallelTracePlayer) keepStallInfo(gpuID int, nextGPU int, roundID int, layer *triosim.Layer) {
	stallInfo := []int{gpuID, nextGPU, roundID, layer.ID}
	p.stallInfos = append(p.stallInfos, stallInfo)
}

func (p *PipeParallelTracePlayer) keepRoundStallInfo(gpuID int, nextGPU int, roundID int, layer *triosim.Layer) {
	roundstallInfos := []int{gpuID, nextGPU, roundID, layer.ID}
	p.roundstallInfos = append(p.roundstallInfos, roundstallInfos)
	// fmt.Println("keepRoundStallInfo", p.roundstallInfos)
}
func (p *PipeParallelTracePlayer) removeRoundStallInfo(gpuID int, roundID int) {
	newRoundstallInfos := [][]int{}
	for _, info := range p.roundstallInfos {
		if info[0] == gpuID && info[2] == roundID {
			p.scheduleNextPipeline(info[0], info[2])
			fmt.Println("removeRoundStallInfo", info)
			// Skip this entry, effectively removing it
			continue
		}
		newRoundstallInfos = append(newRoundstallInfos, info)
	}

	p.roundstallInfos = newRoundstallInfos
}

func (p *PipeParallelTracePlayer) handleNextStep(gpuID int, nextGPU int, roundID int, layer *triosim.Layer) {
	preRoundID := 0
	if p.pipeAction[roundID] == "dobackward" {
		preRoundID = roundID + 1
		if preRoundID < p.batchSize/p.microBatchSize {
			layerIndex := p.memoryRegions[gpuID].pipeComputingLayerIndex[preRoundID] - 1
			layerID := p.trace[layerIndex].ID
			if layerID < p.memoryRegions[gpuID].finalbackwardlayerid {
				p.keepRoundStallInfo(gpuID, nextGPU, roundID, layer)
				return
			}
		}
	} else {
		preRoundID = roundID - 1
		if preRoundID >= 0 {
			layerIndex := p.memoryRegions[gpuID].pipeComputingLayerIndex[preRoundID] - 1
			layerID := p.trace[layerIndex].ID
			if layerID < p.memoryRegions[gpuID].finalforwardLayerID {
				p.keepRoundStallInfo(gpuID, nextGPU, roundID, layer)
				return
			}
		}
	}
	p.processNextPipeline(gpuID, nextGPU, roundID, layer)
}

func (p *PipeParallelTracePlayer) processNextPipeline(gpuID int, nextGPU int, roundID int, layer *triosim.Layer) {
	if nextGPU == gpuID {
		p.scheduleNextPipeline(nextGPU, roundID)
		return
	}

	if len(p.roundstallInfos) > 0 {
		p.handleRoundStallInfos(gpuID, roundID)
	}

	if nextGPU == -1 {
		return //backward is done
	}

	if len(p.stallInfos) > 0 && !p.stall {
		for item := range p.stallInfos {
			stallInfo := p.stallInfos[item]
			p.scheduleNextPipeline(stallInfo[0], stallInfo[2])
		}
		p.stallInfos = nil
	}

	purpose := p.getMessagePurpose(gpuID, roundID)
	err := p.msgPkgToSend(p.memoryRegions[gpuID].Name, p.memoryRegions[nextGPU].Name,
		layer.Outputs, nextGPU, purpose, roundID)
	if !err {
		return
	}
}

func (p *PipeParallelTracePlayer) getMessagePurpose(gpuID, roundID int) string {
	if gpuID == 0 && p.pipeAction[roundID] == "doforward" &&
		p.numRound < (p.batchSize/p.microBatchSize-1) {
		return "nextRound"
	}
	return "nextGPU"
}

func (p *PipeParallelTracePlayer) handleRoundStallInfos(gpuID int, roundID int) {
	layerIndex := p.memoryRegions[gpuID].pipeComputingLayerIndex[roundID] - 1
	layerID := p.trace[layerIndex].ID
	if p.pipeAction[roundID] == "dobackward" {
		nextRoundID := roundID - 1
		if nextRoundID >= 0 {
			if layerID == p.memoryRegions[gpuID].finalbackwardlayerid {
				p.removeRoundStallInfo(gpuID, nextRoundID)
			}
		}
	} else {
		nextRoundID := roundID + 1
		if nextRoundID < p.batchSize/p.microBatchSize {
			if layerID == p.memoryRegions[gpuID].finalforwardLayerID {
				p.removeRoundStallInfo(gpuID, nextRoundID)
			}
		}
	}
}

func (p *PipeParallelTracePlayer) updateMaxFetchComputeIndex(gpuID int, nextGPU int, roundID int) {
	region := p.memoryRegions[gpuID]
	maxIndex := region.pipeFetchingLayerIndex[roundID]
	if region.pipeComputingLayerIndex[roundID] > maxIndex {
		maxIndex = region.pipeComputingLayerIndex[roundID]
	}
	//update the fetching and computing index to the the max value when update GPU each time
	p.memoryRegions[nextGPU].pipeFetchingLayerIndex[roundID] = maxIndex
	p.memoryRegions[nextGPU].pipeComputingLayerIndex[roundID] = maxIndex
	//in fact, computingLayerIndex should be always equal to fetchingLayerIndex
}

func (p *PipeParallelTracePlayer) NextRoundPipelineStart() {
	fmt.Println("--------------------NextRoundPipelineStart")
	p.numRound++
	p.memoryRegions[0].pipeFetchingLayerIndex[p.numRound] = 0
	p.memoryRegions[0].pipeComputingLayerIndex[p.numRound] = 0
	p.pipeAction[p.numRound] = "doforward"
	p.scheduleNextPipeline(0, p.numRound)
}

func (p *PipeParallelTracePlayer) logStepDone(gpuID int, roundID int, currentStep string, layerIndex int) {
	fmt.Println(p.CurrentTime(), ", gpu "+strconv.Itoa(gpuID)+
		", "+currentStep+" at LayerIndex,"+strconv.Itoa(layerIndex)+
		"-----numround"+strconv.Itoa(roundID))
}

func (p *PipeParallelTracePlayer) scheduleNextPipeline(gpuID int, roundID int) {
	p.Schedule(playNextPipelineEvent{
		time:    p.CurrentTime(),
		handler: p,
		gpuID:   gpuID,
		roundID: roundID,
	})
}
