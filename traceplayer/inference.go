// Package traceplayer provides a trace player that plays a trace and
// simulates the execution of the trace.
package traceplayer

import (
	"fmt"
	"reflect"
	"strconv"
	"sync"

	"github.com/sarchlab/triosim"
	"github.com/sarchlab/triosim/networkmodel"
	"github.com/sarchlab/triosim/timemodel"
	"gitlab.com/akita/akita/v3/sim"
)

// A playNextReduceHopEvent triggers the player to continue to play the trace when in Ring AllReduce process.
type playNextReduceHopEvent struct {
	time    sim.VTimeInSec
	handler *InferenceTracePlayer
	gpuID   int
}

// Time returns the time of the event.
func (e playNextReduceHopEvent) Time() sim.VTimeInSec {
	return e.time
}

// Handler returns the handler of the event.
func (e playNextReduceHopEvent) Handler() sim.Handler {
	return e.handler
}

// IsSecondary always returns false.
func (e playNextReduceHopEvent) IsSecondary() bool {
	return false
}

// A playNextReduceEvent triggers the player to continue to play the trace when in Ring AllReduce process.
type playNextReduceEvent struct {
	time    sim.VTimeInSec
	handler *InferenceTracePlayer
	gpuID   int
}

// Time returns the time of the event.
func (e playNextReduceEvent) Time() sim.VTimeInSec {
	return e.time
}

// Handler returns the handler of the event.
func (e playNextReduceEvent) Handler() sim.Handler {
	return e.handler
}

// IsSecondary always returns false.
func (e playNextReduceEvent) IsSecondary() bool {
	return false
}

// A playNextEvent triggers the player to continue to play the trace.
type playNextEvent struct {
	time    sim.VTimeInSec
	handler *InferenceTracePlayer
	gpuID   int
}

// GetGPUID returns the GPUID of the event.
func (e playNextEvent) GetGPUID() int {
	return e.gpuID
}

// Time returns the time of the event.
func (e playNextEvent) Time() sim.VTimeInSec {
	return e.time
}

// Handler returns the handler of the event.
func (e playNextEvent) Handler() sim.Handler {
	return e.handler
}

// IsSecondary always returns false.
func (e playNextEvent) IsSecondary() bool {
	return false
}

// A layerCompletionEvent is triggered when a layer is completed.
type layerCompletionEvent struct {
	time    sim.VTimeInSec
	handler *InferenceTracePlayer
	layer   *triosim.Layer
	gpuID   int
}

// GetGPUID returns the GPUID of the event.
func (e layerCompletionEvent) GetGPUID() int {
	return e.gpuID
}

// Time returns the time of the event.
func (e layerCompletionEvent) Time() sim.VTimeInSec {
	return e.time
}

// Handler returns the handler of the event.
func (e layerCompletionEvent) Handler() sim.Handler {
	return e.handler
}

// IsSecondary always returns false.
func (e layerCompletionEvent) IsSecondary() bool {
	return false
}

// MemoryRegion describes a region of memory that can hold tensors.
type MemoryRegion struct {
	Name            string
	CapacityLimited bool
	Capacity        uint64
	GPUID           int

	Tensors             map[string]triosim.Tensor
	inflightTransfer    []*triosim.TensorMsg
	fetchingLayerIndex  int
	computingLayerIndex int
	doingComputing      bool

	chunks      []triosim.Tensor
	maxIter     int
	sendStep    int
	recvStep    int
	commuAction string
	commulayer  int
	checktensor []QueueEntry
}

// TotalUtilizedBytes returns the total number of bytes used by the tensors in
// the region.
func (r *MemoryRegion) TotalUtilizedBytes() uint64 {
	var total uint64

	for _, t := range r.Tensors {
		total += t.Bytes()
	}

	return total
}

// A InferenceTracePlayer replays the forward path of a trace.
type InferenceTracePlayer struct {
	*sim.ComponentBase

	sim.TimeTeller
	sim.EventScheduler
	timeEstimator timemodel.TimeEstimator

	memoryRegions       []*MemoryRegion
	defaultMemoryRegion *MemoryRegion

	trace         triosim.Trace
	batchSize     int
	reducelayer   int
	ncclAction    string
	sendTofinish  int
	gpuchunks     [][]triosim.Tensor
	scatterstep   int
	gatherstep    int
	updateQueues  map[int]*UpdateQueue
	tokenQueues   map[int]map[int]*TokenQueue
	backupWorkers map[int]int
	networkModel  *networkmodel.PacketSwitchingNetworkModel
	gradientSet   map[triosim.Tensor]bool
}

// NewInferenceTracePlayer creates a new InferenceTracePlayer.
func NewInferenceTracePlayer(
	name string,
	tt sim.TimeTeller,
	es sim.EventScheduler,
	timeEstimator timemodel.TimeEstimator,
) *InferenceTracePlayer {
	p := &InferenceTracePlayer{
		timeEstimator:  timeEstimator,
		TimeTeller:     tt,
		EventScheduler: es,
	}

	p.ComponentBase = sim.NewComponentBase(name)

	return p
}

// AddMemoryRegion adds a memory region to the player.
func (p *InferenceTracePlayer) AddMemoryRegion(
	region *MemoryRegion,
	port sim.Port,
) {
	p.memoryRegions = append(p.memoryRegions, region)
	p.AddPort(region.Name, port)
}

// SetDefaultMemoryRegion sets the default memory region.
func (p *InferenceTracePlayer) SetDefaultMemoryRegion(region *MemoryRegion) {
	p.defaultMemoryRegion = region
}

// Handle function of a InferenceTracePlayer handles events.
func (p *InferenceTracePlayer) Handle(e sim.Event) error {
	switch e := e.(type) {
	case playNextEvent:
		gpuID := e.gpuID
		p.playNext(gpuID)
	case layerCompletionEvent:
		p.completeLayer(e)
	case playNextReduceEvent:
		p.playNextReduce()
	case playNextReduceHopEvent:
		p.playNextReduceHop(e.gpuID)
	default:
		panic("InferenceTracePlayer cannot handle this event type " +
			reflect.TypeOf(e).String())
	}

	return nil
}

// NotifyPortFree function of a InferenceTracePlayer notifies that the
// one port of the component if free.
func (p *InferenceTracePlayer) NotifyPortFree(
	now sim.VTimeInSec,
	port sim.Port,
) {
	msg := port.Retrieve(now)
	msginfo := msg.(*triosim.TensorMsg)
	if msginfo.Purpose == "scatter" || msginfo.Purpose == "gather" {
		p.playNextReduce()
		fmt.Println("recv reduce")
	} else if msginfo.Purpose == "hop" {
		p.playNextReduceHop(msginfo.GPUID)
		fmt.Println("recv hop")
	} else {
		p.playNext(msginfo.GPUID)
	}
}

// NotifyRecv function notifies that the component has received a message.
func (p *InferenceTracePlayer) NotifyRecv(
	now sim.VTimeInSec,
	port sim.Port,
) {
	msg := port.Retrieve(now)
	var gpuID int
	switch msg := msg.(type) {
	case *triosim.TensorMsg:
		p.recvTensorPkg(msg)
		gpuID = msg.GPUID
		if msg.Purpose == "scatter" || msg.Purpose == "gather" || msg.Purpose == "hop" {
			fmt.Println(p.CurrentTime(), ", gpu ", msg.Src.Name(), "to gpu ", msg.Dst.Name(), msg.Purpose)
			p.Schedule(playNextReduceEvent{
				time:    p.CurrentTime(),
				handler: p,
				gpuID:   gpuID,
			})
		} else {
			// fmt.Println(p.CurrentTime(), ", gpu "+strconv.Itoa(gpuID), ", ", msg.Purpose)
			p.Schedule(playNextEvent{
				time:    p.CurrentTime(),
				handler: p,
				gpuID:   gpuID,
			})
		}
	default:
		panic(fmt.Sprintf("Cannot handle message %T", msg))
	}
}

func (p *InferenceTracePlayer) recvTensorPkg(msg *triosim.TensorMsg) {
	p.removeInflightTransfer(msg)
	p.addTensorsToMemRegion(msg.TensorPkg, msg.GPUID, msg.Purpose)
	// fmt.Println(p.CurrentTime(), ", gpu "+strconv.Itoa(msg.GPUID), ", ", msg.Purpose)
	if msg.Purpose == "scatter" || msg.Purpose == "gather" {
		p.sendTofinish--
	}
}

func (p *InferenceTracePlayer) removeInflightTransfer(msg *triosim.TensorMsg) {
	removed := false
	gpuID := msg.GPUID
	for i, m := range p.memoryRegions[gpuID].inflightTransfer {
		if m == msg {
			p.memoryRegions[gpuID].inflightTransfer = append(
				p.memoryRegions[gpuID].inflightTransfer[:i],
				p.memoryRegions[gpuID].inflightTransfer[i+1:]...,
			)
			removed = true
			break
		}
	}
	if !removed {
		panic("Cannot find the message in inflight")
	}
}
func (p *InferenceTracePlayer) SetNetworkModel(networkModel *networkmodel.PacketSwitchingNetworkModel) {
	p.networkModel = networkModel
}

// SetTrace sets the trace to replay by the InferenceTracePlayer.
func (p *InferenceTracePlayer) SetTrace(
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

// KickStart starts the simulation. It will schedule the first playNextEvent.
// The main program should still call engine.run() to run the simulation.
func (p *InferenceTracePlayer) KickStart() {
	if (p.trace == nil) || (len(p.trace) == 0) {
		panic("Trace is not set")
	}

	for _, item := range p.memoryRegions {
		// fmt.Printf("Memory region %s \n", item.Name)
		p.Schedule(playNextEvent{
			time:    p.CurrentTime(),
			handler: p,
			gpuID:   item.GPUID,
		})
	}
}

func (p *InferenceTracePlayer) addTensorsToDefaultMemRegion(
	trace triosim.Trace,
) {
	for _, layer := range trace {
		for _, tensor := range layer.Inputs {
			p.defaultMemoryRegion.Tensors[tensor.ID] = tensor
		}
	}
}

func (p *InferenceTracePlayer) msgPkgToSend(
	srcRegion string,
	dstRegion string,
	tensors []triosim.Tensor,
	gpuID int,
	purpose string,
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
		p.memoryRegions[gpuID].inflightTransfer = append(
			p.memoryRegions[gpuID].inflightTransfer, msg)
	}
	return err == nil
}

// playNext performs the next action that replays the trace
func (p *InferenceTracePlayer) playNext(gpuID int) {
	p.doFetching(gpuID)
	p.doComputing(gpuID)
}

func (p *InferenceTracePlayer) doComputing(gpuID int) {
	if p.memoryRegions[gpuID].doingComputing {
		return
	}

	if p.memoryRegions[gpuID].computingLayerIndex >= len(p.trace) {
		return
	}

	layer := p.trace[p.memoryRegions[gpuID].computingLayerIndex]

	fmt.Println(p.CurrentTime(), ", gpu "+strconv.Itoa(gpuID)+
		", computing check at LayerIndex,"+strconv.Itoa(p.memoryRegions[gpuID].computingLayerIndex))

	_, needsMoving := p.nextTensorPkgToMove(layer, gpuID)
	if needsMoving {
		return
	}

	fmt.Println(p.CurrentTime(), ", gpu "+strconv.Itoa(gpuID)+
		", computing start at LayerIndex,"+strconv.Itoa(p.memoryRegions[gpuID].computingLayerIndex))

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
	evt := layerCompletionEvent{
		time:    now + sim.VTimeInSec(output.TimeInSec),
		handler: p,
		layer:   layer,
		gpuID:   gpuID,
	}
	p.Schedule(evt)

	p.memoryRegions[gpuID].doingComputing = true
	p.memoryRegions[gpuID].computingLayerIndex++
}

func (p *InferenceTracePlayer) completeLayer(e sim.Event) {
	evt := e.(layerCompletionEvent)
	layer := evt.layer
	gpuID := evt.gpuID

	p.addTensorsToMemRegion(layer.Outputs, gpuID, "complete")

	p.memoryRegions[gpuID].doingComputing = false

	fmt.Println(p.CurrentTime(), ", gpu "+strconv.Itoa(gpuID)+
		", computing done at LayerIndex,"+strconv.Itoa(p.memoryRegions[gpuID].computingLayerIndex-1))

	p.Schedule(playNextEvent{
		time:    p.CurrentTime(),
		handler: p,
		gpuID:   gpuID,
	})
}

func (p *InferenceTracePlayer) doFetching(gpuID int) {
	if len(p.memoryRegions[gpuID].inflightTransfer) > 0 {
		return
	}

	if p.memoryRegions[gpuID].fetchingLayerIndex >= len(p.trace) {
		return
	}

	if p.memoryRegions[gpuID].fetchingLayerIndex < p.memoryRegions[gpuID].computingLayerIndex {
		p.memoryRegions[gpuID].fetchingLayerIndex++
		p.Schedule(playNextEvent{
			time:    p.CurrentTime(),
			handler: p,
			gpuID:   gpuID,
		})
		return
	}

	// fmt.Println(p.CurrentTime(), ", gpu "+strconv.Itoa(gpuID)+
	// 	", fetching check at LayerIndex,"+strconv.Itoa(p.memoryRegions[gpuID].fetchingLayerIndex))
	if p.allTensorsOfLayerAreAvailable(p.trace[p.memoryRegions[gpuID].fetchingLayerIndex], gpuID) {
		p.memoryRegions[gpuID].fetchingLayerIndex++
		p.doFetching(gpuID)
		return
	}

	tensors, needsMoving := p.nextTensorPkgToMove(
		p.trace[p.memoryRegions[gpuID].fetchingLayerIndex], gpuID)
	if !p.addTensorsToMemRegion(tensors, gpuID, "allocate") {
		return
	}

	p.addTensorsToMemRegion(p.trace[p.memoryRegions[gpuID].fetchingLayerIndex].Outputs, gpuID, "allocate")
	if !needsMoving {
		p.memoryRegions[gpuID].fetchingLayerIndex++
		p.Schedule(playNextEvent{
			time:    p.CurrentTime(),
			handler: p,
			gpuID:   gpuID,
		})
		return
	}

	// fmt.Println(p.CurrentTime(), ", gpu, "+strconv.Itoa(gpuID)+
	// 	", fetching start at LayerIndex,"+strconv.Itoa(p.memoryRegions[gpuID].fetchingLayerIndex))
	err := p.msgPkgToSend(p.defaultMemoryRegion.Name, p.memoryRegions[gpuID].Name, tensors, gpuID, "fetch")
	if !err {
		return
	}
}

func (p *InferenceTracePlayer) nextTensorPkgToMove(
	layer *triosim.Layer,
	gpuID int,
) (tensors []triosim.Tensor, exist bool) {
	for _, tensor := range layer.Inputs {
		// if tensor.Category != triosim.Gradient && !p.isTensorReady(tensor, gpuID) {
		if !p.isTensorReady(tensor, gpuID) {
			tensors = append(tensors, tensor)
		}
	}

	if tensors != nil {
		return tensors, true
	}

	return nil, false
}

func (p *InferenceTracePlayer) isTensorInGradientSet(tensor triosim.Tensor) bool {
	_, exists := p.gradientSet[tensor]
	return exists
}

func (p *InferenceTracePlayer) isTensorReady(
	tensor triosim.Tensor,
	gpuID int,
) bool {
	region := p.memoryRegions[gpuID]

	_, ok := region.Tensors[tensor.ID]

	return ok
}

func (p *InferenceTracePlayer) allTensorsOfLayerAreAvailable(
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

func (p *InferenceTracePlayer) checkSpaceForTensors(
	tensor []triosim.Tensor,
	gpuID int,
	purpose string,
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

	if purpose == "reduce" || purpose == "scatter" || purpose == "gather" || purpose == "hop" {
		reduceLayerIndex := p.reducelayer
		if purpose == "hop" {
			reduceLayerIndex = p.memoryRegions[gpuID].commulayer
		}
		p.removeTensorFromMemRegion(gpuID, true, reduceLayerIndex)
	} else {
		p.removeTensorFromMemRegion(gpuID, false, 0)
	}

	totalBytes = region.TotalUtilizedBytes() + totalTensorBytes
	if totalBytes <= region.Capacity {
		return true
	}

	fmt.Println("region is full after deleting")
	return false
}

func (p *InferenceTracePlayer) addTensorsToMemRegion(
	tensors []triosim.Tensor,
	gpuID int,
	purpose string,
) bool {
	region := p.memoryRegions[gpuID]
	tensorsCheck := p.filterTensors(tensors, gpuID, purpose)
	if len(tensorsCheck) == 0 {
		return true
	}
	if p.checkSpaceForTensors(tensorsCheck, gpuID, purpose) {
		var Status triosim.TensorMemoryStatus
		switch purpose {
		case "complete":
			Status = triosim.TensorMemoryStatusUsed
		case "fetch":
			Status = triosim.TensorMemoryStatusToBeUsed
		case "allocate":
			Status = triosim.TensorMemoryStatusAllocated
		case "reduce":
			Status = triosim.TensorMemoryStatusAllocated
		default: //gather, scatter, hop
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

func (p *InferenceTracePlayer) filterTensors(
	tensors []triosim.Tensor,
	gpuID int,
	purpose string,
) []triosim.Tensor {
	region := p.memoryRegions[gpuID]
	var tensorsCheck []triosim.Tensor

	switch purpose {
	case "scatter", "gather", "reduce", "hop":
		tensorsCheck = p.handleScatterGatherReduce(tensors, region, purpose)
	case "fetch":
		tensorsCheck = p.handleFetch(tensors, region)
	default: //"allocate", "complete"
		tensorsCheck = p.handleAllocateComplete(tensors, region, purpose)
	}

	return tensorsCheck
}

func (p *InferenceTracePlayer) handleScatterGatherReduce(
	tensors []triosim.Tensor,
	region *MemoryRegion,
	purpose string,
) []triosim.Tensor {
	tensorsCheck := make([]triosim.Tensor, 0)
	for _, tensor := range tensors {
		tensor.ID = tensor.ID + "chunk" + strconv.Itoa(tensor.ChunkID)
		dstTensor, found := region.Tensors[tensor.ID]
		if found {
			if purpose == "scatter" || purpose == "hop" {
				chunksToCombine := []triosim.Tensor{dstTensor, tensor}
				// tensor = combineChunks(chunksToCombine)
				tensor = avgChunks(chunksToCombine)
			}

			delete(region.Tensors, tensor.ID)
		}

		tensorsCheck = append(tensorsCheck, tensor)
	}

	return tensorsCheck
}

func (p *InferenceTracePlayer) handleFetch(
	tensors []triosim.Tensor,
	region *MemoryRegion,
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

func (p *InferenceTracePlayer) handleAllocateComplete(
	tensors []triosim.Tensor,
	region *MemoryRegion,
	purpose string,
) []triosim.Tensor {
	tensorsCheck := make([]triosim.Tensor, 0)
	MemoryStatus := triosim.TensorMemoryStatusAllocated
	if purpose == "complete" {
		MemoryStatus = triosim.TensorMemoryStatusUsed
		tensors = append(tensors, p.trace[region.computingLayerIndex-1].Inputs...)
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

func (p *InferenceTracePlayer) removeTensorFromMemRegion(
	gpuID int,
	forreduce bool,
	reduceLayerIndex int,
) {
	region := p.memoryRegions[gpuID]
	removed := false
	existingTensorIDs := make(map[string]struct{})
	var layer *triosim.Layer
	var LayerTensors []triosim.Tensor
	if forreduce {
		layer = p.trace[reduceLayerIndex]
		LayerTensors = layer.Outputs
	} else {
		if region.computingLayerIndex == len(p.trace) {
			fmt.Println("the region is full when the last layer, do not need to store it in the memory.")
			return
		}

		layer = p.trace[region.computingLayerIndex]
		LayerTensors = append(layer.Outputs, layer.Inputs...)
	}

	for _, layertensor := range LayerTensors {
		existingTensorIDs[layertensor.ID] = struct{}{}
	}

	for _, t := range region.Tensors {
		if _, exists := existingTensorIDs[t.ID]; !exists {
			if forreduce || t.MemoryStatus == triosim.TensorMemoryStatusUsed {
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

func (p *InferenceTracePlayer) getGradientSet() map[triosim.Tensor]bool {
	if p.gradientSet == nil {
		p.gradientSet = make(map[triosim.Tensor]bool) // Replace Tensor with the correct type
	}
	var size uint64
	for _, layer := range p.trace {
		if layer.Name == "aten::_foreach_addcdiv_" {
			for _, tensor := range layer.Inputs {
				// p.gradientSet = append(p.gradientSet, tensor)
				p.gradientSet[tensor] = false
				size += tensor.Bytes()
			}
		}
	}
	fmt.Println("size here", size)
	size = 0
	for _, layer := range p.trace {
		if layer.Name == "aten::_foreach_add_" || layer.Name == "aten::_foreach_lerp_" {
			for _, tensor := range layer.Inputs {
				if _, exists := p.gradientSet[tensor]; exists {
					delete(p.gradientSet, tensor) // Deleting tensor from set if it exists
					size += tensor.Bytes()
				}
			}
		}
	}
	fmt.Println("size delete here", size)
	return p.gradientSet
}

func (p *InferenceTracePlayer) AllReduceStart() {
	if (p.trace == nil) || (len(p.trace) == 0) {
		panic("Trace is not set")
	}

	p.gradientSet = p.getGradientSet()
	for _, item := range p.memoryRegions {
		item.Tensors = make(map[string]triosim.Tensor)
	}

	p.ncclAction = "nextlayer"

	p.Schedule(playNextReduceEvent{
		time:    p.CurrentTime(),
		handler: p,
		gpuID:   0,
	})
}

// playNextReduce performs the next action that replays the trace when in Ring Allreduce process
func (p *InferenceTracePlayer) playNextReduce() {
	p.doAllReduce()
	p.doScatter()
	p.doAllgather()
}

func (p *InferenceTracePlayer) nextReduceTensorPkgToMove(
	layer *triosim.Layer,
) (tensors []triosim.Tensor, exist bool) {
	tensorsize := 0
	for _, tensor := range layer.Inputs {
		if p.isTensorInGradientSet(tensor) {
			tensors = append(tensors, tensor)
		}
		tensorsize += int(tensor.Bytes())
	}
	if tensors != nil {
		return tensors, true
	}

	return nil, false
}

func (p *InferenceTracePlayer) doAllReduce() {
	if p.ncclAction != "nextlayer" {
		return
	}

	for i := 0; i < len(p.memoryRegions)-1; i++ {
		if len(p.memoryRegions[i].inflightTransfer) > 0 {
			return
		}
	}

	if p.reducelayer >= len(p.trace) {
		return
	}

	if p.trace[p.reducelayer].Name != "aten::_foreach_addcdiv_" {
		p.reducelayer++
		p.Schedule(playNextReduceEvent{
			time:    p.CurrentTime(),
			handler: p,
			gpuID:   0,
		})
		return
	} //only do broadcast and reduce for this finallayer

	// data chunking
	tensors, needsMoving :=
		p.nextReduceTensorPkgToMove(p.trace[p.reducelayer])
	if !needsMoving {
		// fmt.Println("this layer has no tensor to reduce", p.reducelayer)
		p.reducelayer++
		p.Schedule(playNextReduceEvent{
			time:    p.CurrentTime(),
			handler: p,
			gpuID:   0,
		})
		return
	}
	// fmt.Println("----tensors", tensors)
	numChunks := len(p.memoryRegions) - 1
	chunks := divideTensor(tensors, numChunks)
	gpuChunks := make([][]triosim.Tensor, numChunks)
	for i := range gpuChunks {
		newChunk := make([]triosim.Tensor, len(chunks))
		copy(newChunk, chunks)
		gpuChunks[i] = newChunk
		p.addTensorsToMemRegion(newChunk, i, "reduce")
	}

	p.gpuchunks = gpuChunks
	p.ncclAction = "scatter"
}

func (p *InferenceTracePlayer) doScatter() {
	if p.ncclAction != "scatter" {
		return
	}

	if p.sendTofinish > 0 {
		return
	}

	if p.scatterstep >= len(p.gpuchunks)-1 {
		return
	}

	chunks := p.gpuchunks
	step := p.scatterstep
	for gpuID := 0; gpuID < len(p.memoryRegions)-1; gpuID++ {
		gpuIDToSend := ((gpuID+1)%len(chunks) + len(chunks)) % len(chunks)
		gpuIDPreSend := ((gpuID-1)%len(chunks) + len(chunks)) % len(chunks)
		idsend := ((gpuID-step)%len(chunks) + len(chunks)) % len(chunks)
		idrecv := ((idsend-1)%len(chunks) + len(chunks)) % len(chunks)
		idpresend := ((gpuID-step-1)%len(chunks) + len(chunks)) % len(chunks)

		tensor := chunks[gpuID][idsend]
		tensor.ID = chunks[gpuIDToSend][idsend].ID
		tensors := make([]triosim.Tensor, 0)
		tensors = append(tensors, tensor)
		if !p.checkSpaceForTensors(tensors, gpuIDToSend, "scatter") {
			return
		}

		fmt.Println(p.CurrentTime(), ", gpu "+strconv.Itoa(gpuID)+" to gpu"+strconv.Itoa(gpuIDToSend)+
			" send chunk "+strconv.Itoa(idsend)+", scatter at LayerIndex,"+strconv.Itoa(p.reducelayer))
		//reduce process
		chunksToCombine := []triosim.Tensor{chunks[gpuID][idrecv], chunks[gpuIDPreSend][idpresend]}
		// combinedTensor := combineChunks(chunksToCombine)
		combinedTensor := avgChunks(chunksToCombine)
		chunks[gpuID][idrecv] = combinedTensor
		tensor.MemoryStatus = triosim.TensorMemoryStatusAvailable
		tensors = make([]triosim.Tensor, 0)
		tensors = append(tensors, tensor)
		err := p.msgPkgToSend(p.memoryRegions[gpuID].Name, p.memoryRegions[gpuIDToSend].Name, tensors, gpuIDToSend, "scatter")
		if !err {
			return
		}
	}

	p.sendTofinish = len(p.memoryRegions) - 1
	p.scatterstep++
	p.gpuchunks = chunks

	if p.scatterstep == len(p.gpuchunks)-1 {
		p.ncclAction = "allgather"
		p.scatterstep = 0
	}
}

func (p *InferenceTracePlayer) doAllgather() {
	if p.ncclAction != "allgather" {
		return
	}

	if p.sendTofinish > 0 {
		return
	}

	if p.gatherstep >= len(p.gpuchunks)-1 {
		return
	}

	step := p.gatherstep
	chunks := p.gpuchunks
	for gpuID := 0; gpuID < len(p.memoryRegions)-1; gpuID++ {
		gpuIDToSend := ((gpuID+1)%len(chunks) + len(chunks)) % len(chunks)
		gpuIDPreSend := ((gpuID-1)%len(chunks) + len(chunks)) % len(chunks)
		idsend := ((gpuID+1-step)%len(chunks) + len(chunks)) % len(chunks)
		idrecv := ((gpuID-step)%len(chunks) + len(chunks)) % len(chunks)
		idpresend := idrecv

		tensor := chunks[gpuID][idsend]
		tensor.ID = chunks[gpuIDToSend][idsend].ID
		tensors := make([]triosim.Tensor, 0)
		tensors = append(tensors, tensor)
		if !p.checkSpaceForTensors(tensors, gpuIDToSend, "gather") {
			return
		}

		fmt.Println(p.CurrentTime(), ", gpu "+strconv.Itoa(gpuID)+" to gpu"+strconv.Itoa(gpuIDToSend)+
			" send chunk "+strconv.Itoa(idsend)+", gather at LayerIndex,"+strconv.Itoa(p.reducelayer))
		tensorID := chunks[gpuID][idrecv].ID
		chunks[gpuID][idrecv] = chunks[gpuIDPreSend][idpresend]
		chunks[gpuID][idrecv].ID = tensorID
		tensor.MemoryStatus = triosim.TensorMemoryStatusAvailable
		tensors = make([]triosim.Tensor, 0)
		tensors = append(tensors, tensor)
		err := p.msgPkgToSend(p.memoryRegions[gpuID].Name, p.memoryRegions[gpuIDToSend].Name, tensors, gpuIDToSend, "gather")
		if !err {
			return
		}
	}

	p.sendTofinish = len(p.memoryRegions) - 1
	p.gatherstep++
	p.gpuchunks = chunks
	if p.gatherstep == len(p.gpuchunks)-1 {
		p.ncclAction = "nextlayer"
		p.gatherstep = 0
		p.reducelayer++
	}
}

func divideTensor(
	tensors []triosim.Tensor,
	numChunks int,
) []triosim.Tensor {
	totalElements := 0
	for _, tensor := range tensors {
		totalElements += int(tensor.Bytes())
	}
	chunkSize := totalElements / numChunks
	remainder := totalElements % numChunks

	chunks := make([]triosim.Tensor, numChunks)
	startIndex := 0
	for i := 0; i < numChunks; i++ {
		chunk := triosim.Tensor{
			ID:           tensors[0].ID,
			ChunkID:      i,
			GPUID:        tensors[0].GPUID,
			MemoryStatus: tensors[0].MemoryStatus,
			Index:        tensors[0].Index,
			Size:         0,
			Category:     triosim.Gradient,
		}
		chunk.Size = chunkSize
		if i == 0 {
			chunk.Size += remainder
		}
		startIndex += chunkSize
		chunks[i] = chunk
	}
	return chunks
}

func combineChunks(chunks []triosim.Tensor) triosim.Tensor {
	firstChunk := chunks[0]
	combinedTensor := triosim.Tensor{
		ID:           firstChunk.ID,
		ChunkID:      firstChunk.ChunkID,
		GPUID:        firstChunk.GPUID,
		MemoryStatus: firstChunk.MemoryStatus,
		Index:        firstChunk.Index,
		Size:         0,
		Category:     triosim.Gradient,
	}

	for _, chunk := range chunks {
		combinedTensor.Size += chunk.Size
	}
	return combinedTensor
}

func avgChunks(chunks []triosim.Tensor) triosim.Tensor {
	firstChunk := chunks[0]
	avgTensor := triosim.Tensor{
		ID:           firstChunk.ID,
		ChunkID:      firstChunk.ChunkID,
		GPUID:        firstChunk.GPUID,
		MemoryStatus: firstChunk.MemoryStatus,
		Index:        firstChunk.Index,
		Size:         0,
		Category:     triosim.Gradient,
	}
	for _, chunk := range chunks {
		avgTensor.Size += chunk.Size
	}
	avgTensor.Size /= len(chunks)
	return avgTensor
}

func (p *InferenceTracePlayer) initHop(backupNum int) {
	p.updateQueues = make(map[int]*UpdateQueue)
	for i := 0; i < len(p.memoryRegions)-1; i++ {
		p.updateQueues[i] = &UpdateQueue{queue: []QueueEntry{}}
	} //init update queue
	p.tokenQueues = make(map[int]map[int]*TokenQueue)
	//add backup workers version
	p.backupWorkers = make(map[int]int)
	for i := 0; i < len(p.memoryRegions)-1; i++ {
		p.backupWorkers[i] = backupNum //only one backup worker now
	}
	maxIg := len(p.memoryRegions) //check it later
	for gpuID := 0; gpuID < len(p.memoryRegions)-1; gpuID++ {
		for _, neighbor := range p.networkModel.FindNeighbor(p.memoryRegions[gpuID].Name, "out") {
			p.InitializeTokenQueue(neighbor, gpuID)
			p.InitializeTokenQueue(gpuID, neighbor)
		}
	}
	for gpuID := 0; gpuID < len(p.memoryRegions)-1; gpuID++ {
		for _, neighbor := range p.networkModel.FindNeighbor(p.memoryRegions[gpuID].Name, "in") {
			tokens := make([]int, maxIg-1)
			p.tokenQueues[gpuID][neighbor].Enqueue(tokens)
		}
	}
}
func (p *InferenceTracePlayer) HopAllReduceStart(backupNum int) {
	if (p.trace == nil) || (len(p.trace) == 0) {
		panic("Trace is not set")
	}

	p.gradientSet = p.getGradientSet()
	for _, item := range p.memoryRegions {
		item.Tensors = make(map[string]triosim.Tensor)
	}

	p.initHop(backupNum)
	for _, item := range p.memoryRegions {
		fmt.Printf("Memory region %s \n", item.Name)
		item.commuAction = "nextlayer"
		p.Schedule(playNextReduceHopEvent{
			time:    p.CurrentTime(),
			handler: p,
			gpuID:   item.GPUID,
		})
	}
}

// playNextReduceHop performs the next action that replays the trace when in Ring Allreduce process
func (p *InferenceTracePlayer) playNextReduceHop(gpuID int) {
	p.doHopNextLayer(gpuID)
	p.doSendHopBackup(gpuID) //ring-based or other graph backup workers version
	p.doRecvHopBackup(gpuID)
}
func (p *InferenceTracePlayer) doHopNextLayer(gpuID int) {
	fmt.Println("doAllReduceHop", p.memoryRegions[gpuID].commuAction, gpuID)
	if p.memoryRegions[gpuID].commuAction != "nextlayer" {
		return
	}

	if p.memoryRegions[gpuID].commulayer >= len(p.trace) {
		fmt.Println("finish all the layers", gpuID)
		return
	}
	if p.memoryRegions[gpuID].recvStep > p.memoryRegions[gpuID].maxIter ||
		p.memoryRegions[gpuID].sendStep > p.memoryRegions[gpuID].maxIter {
		fmt.Println("finish all the iter", gpuID)
		return
	}

	if p.trace[p.memoryRegions[gpuID].commulayer].Name != "aten::_foreach_addcdiv_" {
		p.memoryRegions[gpuID].commulayer++
		p.Schedule(playNextReduceHopEvent{
			time:    p.CurrentTime(),
			handler: p,
			gpuID:   gpuID,
		})
		return
	} //only do broadcast and reduce for this finallayer

	// data chunking
	tensors, needsMoving :=
		p.nextReduceTensorPkgToMove(p.trace[p.memoryRegions[gpuID].commulayer])
	if !needsMoving {
		// fmt.Println("this layer has no tensor to reduce", p.reducelayer)
		p.memoryRegions[gpuID].commulayer++
		p.Schedule(playNextReduceHopEvent{
			time:    p.CurrentTime(),
			handler: p,
			gpuID:   gpuID,
		})
		return
	}
	combinedTensor := combineChunks(tensors)
	combinedTensors := []triosim.Tensor{}
	combinedTensors = append(combinedTensors, combinedTensor)
	p.addTensorsToMemRegion(combinedTensors, gpuID, "reduce")

	p.memoryRegions[gpuID].chunks = combinedTensors
	p.memoryRegions[gpuID].commuAction = "send"
	maxIter := len(p.memoryRegions) - 2 //set it later
	p.memoryRegions[gpuID].maxIter = maxIter
}

func (p *InferenceTracePlayer) doSendHopBackup(gpuID int) {
	fmt.Println("doSendHopbackup", p.memoryRegions[gpuID].sendStep, p.memoryRegions[gpuID].commuAction, gpuID)
	if p.memoryRegions[gpuID].commuAction != "send" {
		return
	}

	chunks := p.memoryRegions[gpuID].chunks
	if p.memoryRegions[gpuID].sendStep < p.memoryRegions[gpuID].maxIter { //intsert token
		for _, neighbor := range p.networkModel.FindNeighbor(p.memoryRegions[gpuID].Name, "in") {
			p.tokenQueues[gpuID][neighbor].Enqueue([]int{p.memoryRegions[gpuID].sendStep})
			p.handleOutNeighbors(gpuID, chunks)
		}
	}
	p.memoryRegions[gpuID].sendStep++
	p.memoryRegions[gpuID].commuAction = "recv" //update queue done and send done
}

func (p *InferenceTracePlayer) handleOutNeighbors(gpuID int, chunks []triosim.Tensor) {
	//end insert tokenQueues and send start
	neighbors := p.networkModel.FindNeighbor(p.memoryRegions[gpuID].Name, "out")
	fmt.Println("neighbors", neighbors)
	for i, neighbor := range neighbors {
		tensors := make([]triosim.Tensor, len(chunks)) // Create a slice with the same length as chunks
		copy(tensors, chunks)
		for i := range tensors {
			tensors[i].ID = tensors[i].ID + "iter" + strconv.Itoa(p.memoryRegions[gpuID].sendStep) +
				"send" + strconv.Itoa(gpuID) + "recv" + strconv.Itoa(neighbor)
			fmt.Println("tensor id", tensors[i].ID)
		}
		if !p.checkSpaceForTensors(tensors, neighbor, "hop") {
			return
		}
		fmt.Println(p.CurrentTime(), ", gpu "+strconv.Itoa(gpuID)+" to gpu"+strconv.Itoa(neighbor)+
			" send chunk "+", send at LayerIndex,"+strconv.Itoa(p.memoryRegions[gpuID].commulayer))
		if p.backupWorkers[gpuID] == 0 { // check it later test for 1 backup worker
			err := p.msgPkgToSend(p.memoryRegions[gpuID].Name, p.memoryRegions[neighbor].Name, tensors, neighbor, "hop")
			if !err {
				return
			}
			entry := QueueEntry{
				iteration: p.memoryRegions[gpuID].sendStep,
				senderID:  gpuID,
				params:    tensors,
			}
			p.updateQueues[neighbor].Enqueue(entry, p.memoryRegions[gpuID].sendStep, gpuID)
			fmt.Println("enqueue p1", neighbor, len(p.updateQueues[neighbor].queue))
		} else {
			if i != len(neighbors)-1 { //len(neighbors)-1
				err := p.msgPkgToSend(p.memoryRegions[gpuID].Name, p.memoryRegions[neighbor].Name, tensors, neighbor, "hop")
				if !err {
					return
				}
				entry := QueueEntry{
					iteration: p.memoryRegions[gpuID].sendStep,
					senderID:  gpuID,
					params:    tensors,
				}
				p.updateQueues[neighbor].Enqueue(entry, p.memoryRegions[gpuID].sendStep, gpuID)
				fmt.Println("enqueue p1", neighbor, len(p.updateQueues[neighbor].queue))
			} else {
				entry := QueueEntry{
					iteration: p.memoryRegions[gpuID].sendStep,
					senderID:  gpuID,
					params:    tensors,
				}
				p.updateQueues[neighbor].Enqueue(entry, p.memoryRegions[gpuID].sendStep, gpuID)
				fmt.Println("enqueue p1", neighbor, len(p.updateQueues[neighbor].queue))
			}
		}
	}
}

func (p *InferenceTracePlayer) doRecvHopBackup(gpuID int) {
	fmt.Println("doRecvHopBackup", p.memoryRegions[gpuID].recvStep, p.memoryRegions[gpuID].commuAction, gpuID)
	if p.memoryRegions[gpuID].commuAction != "recv" {
		return
	}
	recvDone := false
	if p.memoryRegions[gpuID].recvStep < p.memoryRegions[gpuID].maxIter &&
		p.memoryRegions[gpuID].recvStep <= p.memoryRegions[gpuID].sendStep {
		recvDone = p.processRecvStep(recvDone, gpuID)
	}
	if recvDone {
		p.memoryRegions[gpuID].recvStep++
		p.memoryRegions[gpuID].commuAction = "send"
		fmt.Println("finish recv", p.memoryRegions[gpuID].recvStep, gpuID)
		if p.memoryRegions[gpuID].recvStep == p.memoryRegions[gpuID].maxIter {
			p.memoryRegions[gpuID].commuAction = "nextlayer"
			fmt.Println("finish Recv")
			p.memoryRegions[gpuID].commulayer++
		}
	} else {
		fmt.Println("recv not do", p.memoryRegions[gpuID].recvStep, gpuID)
	}
}
func (p *InferenceTracePlayer) processRecvStep(recvDone bool, gpuID int) bool {
	neighbors := p.networkModel.FindNeighbor(p.memoryRegions[gpuID].Name, "in")
	for i := 0; i < len(neighbors); i++ {
		if i != 0 {
			continue
		} //check it later
		//Compute gradients,grads =Compute(xk,i,dk,i) and recv start
		fmt.Println("neighbor", neighbors)
		fmt.Println("backupWorkers", p.backupWorkers[gpuID])
		denum := len(neighbors) - p.backupWorkers[gpuID]
		if denum == 0 {
			fmt.Println("test")
		}
		xRecvsP1 := p.updateQueues[gpuID].Peek(denum, p.memoryRegions[gpuID].recvStep)
		if xRecvsP1 == nil {
			fmt.Println("no recv p1", gpuID)
			continue
		} else {
			fmt.Println("peek p1", xRecvsP1)
		}
		// check if recv in the memoryRegions
		if !p.checkRecv(xRecvsP1, gpuID) {
			fmt.Println("recv p1 not in the memoryRegions", gpuID)
			continue
		}
		if p.backupWorkers[gpuID] != 0 {
			denum = p.handleBackupWorkers(gpuID, denum)
		}
		fmt.Println("dequeue p1 before", gpuID, denum, len(p.updateQueues[gpuID].queue))
		xRecvsP1 = p.updateQueues[gpuID].Dequeue(denum, p.memoryRegions[gpuID].recvStep)
		fmt.Println("dequeue p1 after", gpuID, denum, len(p.updateQueues[gpuID].queue), "reducestart", gpuID)
		avgTensor := p.reduceTensor(xRecvsP1)
		p.memoryRegions[gpuID].chunks = avgTensor
		recvDone = true
		//reduce Done and apply done, done is with send every time when recv the msg tensor accually and get a new token
		p.handleTokens(gpuID)
	}
	return recvDone
}

func (p *InferenceTracePlayer) handleTokens(gpuID int) {
	for _, neighbor := range p.networkModel.FindNeighbor(p.memoryRegions[gpuID].Name, "out") {
		tokens := p.tokenQueues[neighbor][gpuID].Dequeue(1) // Consider parameterizing the dequeue count dequeue 1 or |Nin|
		if tokens == nil {
			// Handle the case where there are not enough tokens.
			fmt.Println("Not enough tokens to proceed.")
		}
	}
}

func (p *InferenceTracePlayer) handleBackupWorkers(gpuID int, denum int) int {
	size := 0 // Step 2: Get additional updates remaining in the queue.
	for _, entry := range p.updateQueues[gpuID].queue {
		if entry.iteration == p.memoryRegions[gpuID].recvStep {
			size++
		}
	}

	if size-denum == 0 {
		fmt.Println("no additional recv p2", gpuID)
	} else {
		xRecvsP2 := p.updateQueues[gpuID].Peek(size, p.memoryRegions[gpuID].recvStep)
		if xRecvsP2 == nil {
			fmt.Println("no recv p2", gpuID)
		} else {
			fmt.Println("peek p2", xRecvsP2)
			p.memoryRegions[gpuID].checktensor = xRecvsP2
			if !p.checkRecv(xRecvsP2, gpuID) {
				fmt.Println("recv p2 not in the memoryRegions", gpuID)
			}

			fmt.Println("p2 recv in the memoryRegions", gpuID)
			denum = size
		}
	}
	return denum
}

func (p *InferenceTracePlayer) reduceTensor(xrecv []QueueEntry) []triosim.Tensor {
	// Compute the number of entries in xrecv
	// Reduce and return
	reduceTensors := []triosim.Tensor{}
	for _, entry := range xrecv {
		reduceTensors = append(reduceTensors, entry.params...)
	}
	if len(reduceTensors) == 0 {
		return []triosim.Tensor{}
	}
	avgTensor := avgChunks(reduceTensors)
	return []triosim.Tensor{avgTensor}
}

func (p *InferenceTracePlayer) InitializeTokenQueue(neighbor int, gpuID int) {
	// Check if the top-level map is nil.
	if p.tokenQueues == nil {
		p.tokenQueues = make(map[int]map[int]*TokenQueue)
	}

	// Check if the sub-map for the neighbor is nil and initialize it if needed.
	if p.tokenQueues[neighbor] == nil {
		p.tokenQueues[neighbor] = make(map[int]*TokenQueue)
	}

	// Check if the TokenQueue for the specific gpuID is nil and initialize it.
	if p.tokenQueues[neighbor][gpuID] == nil {
		p.tokenQueues[neighbor][gpuID] = &TokenQueue{queue: []int{}}
	}
}
func (p *InferenceTracePlayer) checkRecv(xRecvs []QueueEntry, gpuID int) bool {
	for _, entry := range xRecvs {
		tensor := entry.params[0]
		region := p.memoryRegions[gpuID]
		_, ok := region.Tensors[tensor.ID+"chunk"+strconv.Itoa(tensor.ChunkID)]
		if !ok {
			return false
		} else {
			fmt.Println("memory status", region.Tensors[tensor.ID+"chunk"+strconv.Itoa(tensor.ChunkID)].MemoryStatus)
			if region.Tensors[tensor.ID+"chunk"+strconv.Itoa(tensor.ChunkID)].MemoryStatus !=
				triosim.TensorMemoryStatusAvailable {
				return false
			} else {
				fmt.Println("memory status", region.Tensors[tensor.ID+"chunk"+strconv.Itoa(tensor.ChunkID)].MemoryStatus)
			}
		}
	}
	return true
}

// Peek retrieves the first `count` items matching the specified step without removing them.
func (uq *UpdateQueue) Peek(count int, step int) []QueueEntry {
	uq.mu.Lock()
	defer uq.mu.Unlock()

	var peekedEntries []QueueEntry
	if count == 0 {
		return peekedEntries
	}
	for _, entry := range uq.queue {
		if entry.iteration == step {
			peekedEntries = append(peekedEntries, entry)
			if len(peekedEntries) == count {
				break
			}
		}
	}
	return peekedEntries
}

// QueueEntry represents an update with tags for iteration and the ID of the sending worker.
type QueueEntry struct {
	iteration int
	senderID  int
	params    []triosim.Tensor
}

// UpdateQueue is a FIFO queue that stores tagged updates.
type UpdateQueue struct {
	queue []QueueEntry
	mu    sync.Mutex
}

// Enqueue adds a new entry to the queue.
func (uq *UpdateQueue) Enqueue(entry QueueEntry, iteration int, gpuID int) {
	uq.mu.Lock()
	defer uq.mu.Unlock()
	// Update the entry with the provided iteration and gpuID before adding to the queue.
	entry.iteration = iteration
	entry.senderID = gpuID
	// Add the entry to the queue.
	uq.queue = append(uq.queue, entry)
}

// Dequeue retrieves and removes up to `count` entries for a given iteration.
func (uq *UpdateQueue) Dequeue(count int, iteration int) []QueueEntry {
	uq.mu.Lock()
	defer uq.mu.Unlock()

	var matchingEntries []QueueEntry
	var remainingEntries []QueueEntry
	if count == 0 {
		return matchingEntries
	}
	for _, entry := range uq.queue {
		if entry.iteration == iteration {
			if len(matchingEntries) < count {
				matchingEntries = append(matchingEntries, entry)
			} else {
				remainingEntries = append(remainingEntries, entry)
			}
		} else {
			remainingEntries = append(remainingEntries, entry)
		}
	}

	// Update the queue to contain only the non-matching entries.
	uq.queue = remainingEntries
	return matchingEntries
}

// DequeueAllMatching retrieves all entries matching a specific iteration.
func (uq *UpdateQueue) DequeueAllMatching(iteration int, senderID int) []QueueEntry {
	uq.mu.Lock()
	defer uq.mu.Unlock()

	var matchingEntries []QueueEntry
	var remainingEntries []QueueEntry

	for _, entry := range uq.queue {
		if entry.iteration == iteration && entry.senderID == senderID {
			matchingEntries = append(matchingEntries, entry)
		} else {
			remainingEntries = append(remainingEntries, entry)
		}
	}

	// Update the queue to only include non-matching entries.
	uq.queue = remainingEntries
	return matchingEntries
}

type TokenQueue struct {
	queue []int
	mu    sync.Mutex
}

func (tq *TokenQueue) Enqueue(tokens []int) {
	tq.mu.Lock()
	tq.queue = append(tq.queue, tokens...)
	tq.mu.Unlock()
}

func (tq *TokenQueue) Dequeue(count int) []int {
	tq.mu.Lock()
	defer tq.mu.Unlock()
	if len(tq.queue) < count {
		return nil
	}
	tokens := tq.queue[:count]
	tq.queue = tq.queue[count:]
	return tokens
}
