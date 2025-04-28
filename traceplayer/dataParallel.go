package traceplayer

import (
	"fmt"
	"reflect"
	"strconv"

	"github.com/sarchlab/triosim"
	"github.com/sarchlab/triosim/networkmodel"
	"github.com/sarchlab/triosim/timemodel"
	"gitlab.com/akita/akita/v3/sim"
)

// A playNextDataEvent triggers the player to continue to play the trace when in Ring AllReduce process.
type playNextDataEvent struct {
	time    sim.VTimeInSec
	handler *DataParallelTracePlayer
	gpuID   int
}

// Time returns the time of the event.
func (e playNextDataEvent) Time() sim.VTimeInSec {
	return e.time
}

// Handler returns the handler of the event.
func (e playNextDataEvent) Handler() sim.Handler {
	return e.handler
}

// IsSecondary always returns false.
func (e playNextDataEvent) IsSecondary() bool {
	return false
}

// A playNextTraceEvent triggers the player to continue to play the trace.
type playNextTraceEvent struct {
	time    sim.VTimeInSec
	handler *DataParallelTracePlayer
	gpuID   int
}

// GetGPUID returns the GPUID of the event.
func (e playNextTraceEvent) GetGPUID() int {
	return e.gpuID
}

// Time returns the time of the event.
func (e playNextTraceEvent) Time() sim.VTimeInSec {
	return e.time
}

// Handler returns the handler of the event.
func (e playNextTraceEvent) Handler() sim.Handler {
	return e.handler
}

// IsSecondary always returns false.
func (e playNextTraceEvent) IsSecondary() bool {
	return false
}

// A layerCompletionDataEvent is triggered when a layer is completed.
type layerCompletionDataEvent struct {
	time    sim.VTimeInSec
	handler *DataParallelTracePlayer
	layer   *triosim.Layer
	gpuID   int
}

// GetGPUID returns the GPUID of the event.
func (e layerCompletionDataEvent) GetGPUID() int {
	return e.gpuID
}

// Time returns the time of the event.
func (e layerCompletionDataEvent) Time() sim.VTimeInSec {
	return e.time
}

// Handler returns the handler of the event.
func (e layerCompletionDataEvent) Handler() sim.Handler {
	return e.handler
}

// IsSecondary always returns false.
func (e layerCompletionDataEvent) IsSecondary() bool {
	return false
}

// DataMemoryRegion describes a region of memory that can hold tensors.
type DataMemoryRegion struct {
	Name            string
	CapacityLimited bool
	Capacity        uint64
	GPUID           int

	Tensors             map[string]triosim.Tensor
	inflightTransfer    []*triosim.TensorMsg
	fetchingLayerIndex  int
	computingLayerIndex int
	doingComputing      bool
}

// TotalUtilizedBytes returns the total number of bytes used by the tensors in
// the region.
func (r *DataMemoryRegion) TotalUtilizedBytes() uint64 {
	var total uint64

	for _, t := range r.Tensors {
		total += t.Bytes()
	}

	return total
}

// A DataParallelTracePlayer replays the forward path of a trace.
type DataParallelTracePlayer struct {
	*sim.ComponentBase

	sim.TimeTeller
	sim.EventScheduler
	timeEstimator timemodel.TimeEstimator

	memoryRegions       []*DataMemoryRegion
	defaultMemoryRegion *DataMemoryRegion

	trace              triosim.Trace
	batchSize          int
	reducelayer        int
	ncclAction         string
	sendTofinish       int
	gpuchunks          [][]triosim.Tensor
	scatterstep        int
	gatherstep         int
	networkModel       *networkmodel.PacketSwitchingNetworkModel
	gradientSet        map[triosim.Tensor]bool
	totalBytes         uint64
	totalreduceTensors []triosim.Tensor
	reducetrace        []triosim.Layer
}

// NewDataParallelTracePlayer creates a new DataParallelTracePlayer.
func NewDataParallelTracePlayer(
	name string,
	tt sim.TimeTeller,
	es sim.EventScheduler,
	timeEstimator timemodel.TimeEstimator,
) *DataParallelTracePlayer {
	p := &DataParallelTracePlayer{
		timeEstimator:  timeEstimator,
		TimeTeller:     tt,
		EventScheduler: es,
	}

	p.ComponentBase = sim.NewComponentBase(name)

	return p
}

// AddMemoryRegion adds a memory region to the player.
func (p *DataParallelTracePlayer) AddMemoryRegion(
	region *DataMemoryRegion,
	port sim.Port,
) {
	p.memoryRegions = append(p.memoryRegions, region)
	p.AddPort(region.Name, port)
}

// SetDefaultMemoryRegion sets the default memory region.
func (p *DataParallelTracePlayer) SetDefaultMemoryRegion(region *DataMemoryRegion) {
	p.defaultMemoryRegion = region
}

// Handle function of a DataParallelTracePlayer handles events.
func (p *DataParallelTracePlayer) Handle(e sim.Event) error {
	switch e := e.(type) {
	case playNextTraceEvent:
		gpuID := e.gpuID
		p.playNext(gpuID)
	case layerCompletionDataEvent:
		p.completeLayer(e)
	case playNextDataEvent:
		p.playNextReduce()
	default:
		panic("DataParallelTracePlayer cannot handle this event type " +
			reflect.TypeOf(e).String())
	}

	return nil
}

// NotifyPortFree function of a DataParallelTracePlayer notifies that the
// one port of the component if free.
func (p *DataParallelTracePlayer) NotifyPortFree(
	now sim.VTimeInSec,
	port sim.Port,
) {
	msg := port.Retrieve(now)
	msginfo := msg.(*triosim.TensorMsg)
	if msginfo.Purpose == "scatter" || msginfo.Purpose == "gather" {
		p.playNextReduce()
		fmt.Println("recv reduce")
	} else {
		p.playNext(msginfo.GPUID)
	}
}

// NotifyRecv function notifies that the component has received a message.
func (p *DataParallelTracePlayer) NotifyRecv(
	now sim.VTimeInSec,
	port sim.Port,
) {
	msg := port.Retrieve(now)
	var gpuID int
	switch msg := msg.(type) {
	case *triosim.TensorMsg:
		p.recvTensorPkg(msg)
		gpuID = msg.GPUID
		if msg.Purpose == "scatter" || msg.Purpose == "gather" {
			fmt.Println(p.CurrentTime(), ", gpu ", msg.Src.Name(), "to gpu ", msg.Dst.Name(), msg.Purpose)
			p.Schedule(playNextDataEvent{
				time:    p.CurrentTime(),
				handler: p,
				gpuID:   gpuID,
			})
		} else {
			// fmt.Println(p.CurrentTime(), ", gpu "+strconv.Itoa(gpuID), ", ", msg.Purpose)
			p.Schedule(playNextTraceEvent{
				time:    p.CurrentTime(),
				handler: p,
				gpuID:   gpuID,
			})
		}
	default:
		panic(fmt.Sprintf("Cannot handle message %T", msg))
	}
}

func (p *DataParallelTracePlayer) recvTensorPkg(msg *triosim.TensorMsg) {
	p.removeInflightTransfer(msg)
	p.addTensorsToMemRegion(msg.TensorPkg, msg.GPUID, msg.Purpose)
	// fmt.Println(p.CurrentTime(), ", gpu "+strconv.Itoa(msg.GPUID), ", ", msg.Purpose)
	if msg.Purpose == "scatter" || msg.Purpose == "gather" {
		p.sendTofinish--
	}
}

func (p *DataParallelTracePlayer) removeInflightTransfer(msg *triosim.TensorMsg) {
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
func (p *DataParallelTracePlayer) SetNetworkModel(networkModel *networkmodel.PacketSwitchingNetworkModel) {
	p.networkModel = networkModel
}

// SetTrace sets the trace to replay by the DataParallelTracePlayer.
func (p *DataParallelTracePlayer) SetTrace(
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
func (p *DataParallelTracePlayer) getGradientSet() map[triosim.Tensor]bool {
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

// KickStart starts the simulation. It will schedule the first playNextEvent.
// The main program should still call engine.run() to run the simulation.
func (p *DataParallelTracePlayer) KickStart() {
	if (p.trace == nil) || (len(p.trace) == 0) {
		panic("Trace is not set")
	}
	p.gradientSet = p.getGradientSet()
	for _, item := range p.memoryRegions {
		// fmt.Printf("Memory region %s \n", item.Name)
		p.Schedule(playNextTraceEvent{
			time:    p.CurrentTime(),
			handler: p,
			gpuID:   item.GPUID,
		})
	}
	p.ncclAction = "nextlayer"
}

func (p *DataParallelTracePlayer) addTensorsToDefaultMemRegion(
	trace triosim.Trace,
) {
	for _, layer := range trace {
		for _, tensor := range layer.Inputs {
			p.defaultMemoryRegion.Tensors[tensor.ID] = tensor
		}
	}
}

func (p *DataParallelTracePlayer) msgPkgToSend(
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
func (p *DataParallelTracePlayer) playNext(gpuID int) {
	p.doFetching(gpuID)
	p.doComputing(gpuID)
}

// playNextReduce performs the next action that replays the trace when in Ring Allreduce process
func (p *DataParallelTracePlayer) playNextReduce() {
	p.doAllReduce()
	p.doScatter()
	p.doAllgather()
}

func (p *DataParallelTracePlayer) doComputing(gpuID int) {
	if p.memoryRegions[gpuID].doingComputing {
		return
	}

	if p.memoryRegions[gpuID].computingLayerIndex >= len(p.trace) {
		return
	}

	layer := p.trace[p.memoryRegions[gpuID].computingLayerIndex]

	fmt.Println(p.CurrentTime(), ", gpu "+strconv.Itoa(gpuID)+
		", computing check at LayerIndex,"+strconv.Itoa(p.memoryRegions[gpuID].computingLayerIndex))

	_, needsMoving, reduceTensors, reduce := p.nextTensorPkgToMove(layer, gpuID)
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
	evt := layerCompletionDataEvent{
		time:    now + sim.VTimeInSec(output.TimeInSec),
		handler: p,
		layer:   layer,
		gpuID:   gpuID,
	}
	p.Schedule(evt)

	p.memoryRegions[gpuID].doingComputing = true
	p.memoryRegions[gpuID].computingLayerIndex++

	if reduce && gpuID == 0 { //only 1 gpu launch the reduce for all gpus
		p.handleReduceProcess(reduceTensors)
	}
}

func (p *DataParallelTracePlayer) handleReduceProcess(reduceTensors []triosim.Tensor) {
	var newReduceTensors []triosim.Tensor
	for _, tensor := range reduceTensors {
		p.gradientSet[tensor] = true
		for key := range p.gradientSet {
			if key.ID > tensor.ID && !p.gradientSet[key] {
				p.gradientSet[key] = true
				newReduceTensors = append(newReduceTensors, key)
			}
		}
	}
	reduceTensors = append(reduceTensors, newReduceTensors...)
	for _, tensor := range reduceTensors {
		p.totalBytes += tensor.Bytes()
	}

	p.totalreduceTensors = append(p.totalreduceTensors, reduceTensors...)
	fmt.Println("reduce size", p.totalBytes, len(p.totalreduceTensors)-1)
	// fmt.Println("reduce tensors", p.totalreduceTensors)
	newlayer := triosim.Layer{
		Inputs: reduceTensors,
	}
	p.reducetrace = append(p.reducetrace, newlayer)
	if p.reducelayer <= len(p.reducetrace) {
		// p.ncclAction = "nextlayer"
		p.Schedule(playNextDataEvent{
			time:    p.CurrentTime(),
			handler: p,
			gpuID:   0,
		})
	}
}

func (p *DataParallelTracePlayer) completeLayer(e sim.Event) {
	evt := e.(layerCompletionDataEvent)
	layer := evt.layer
	gpuID := evt.gpuID

	p.addTensorsToMemRegion(layer.Outputs, gpuID, "complete")

	p.memoryRegions[gpuID].doingComputing = false

	fmt.Println(p.CurrentTime(), ", gpu "+strconv.Itoa(gpuID)+
		", computing done at LayerIndex,"+strconv.Itoa(p.memoryRegions[gpuID].computingLayerIndex-1))

	p.Schedule(playNextTraceEvent{
		time:    p.CurrentTime(),
		handler: p,
		gpuID:   gpuID,
	})
}

func (p *DataParallelTracePlayer) doFetching(gpuID int) {
	if len(p.memoryRegions[gpuID].inflightTransfer) > 0 {
		return
	}

	if p.memoryRegions[gpuID].fetchingLayerIndex >= len(p.trace) {
		return
	}

	if p.memoryRegions[gpuID].fetchingLayerIndex < p.memoryRegions[gpuID].computingLayerIndex {
		p.memoryRegions[gpuID].fetchingLayerIndex++
		p.Schedule(playNextTraceEvent{
			time:    p.CurrentTime(),
			handler: p,
			gpuID:   gpuID,
		})
		return
	}

	fmt.Println(p.CurrentTime(), ", gpu "+strconv.Itoa(gpuID)+
		", fetching check at LayerIndex,"+strconv.Itoa(p.memoryRegions[gpuID].fetchingLayerIndex))
	if p.allTensorsOfLayerAreAvailable(p.trace[p.memoryRegions[gpuID].fetchingLayerIndex], gpuID) {
		p.memoryRegions[gpuID].fetchingLayerIndex++
		p.doFetching(gpuID)
		return
	}

	tensors, needsMoving, _, _ := p.nextTensorPkgToMove(
		p.trace[p.memoryRegions[gpuID].fetchingLayerIndex], gpuID)

	if !p.addTensorsToMemRegion(tensors, gpuID, "allocate") {
		return
	}

	p.addTensorsToMemRegion(p.trace[p.memoryRegions[gpuID].fetchingLayerIndex].Outputs, gpuID, "allocate")
	if !needsMoving {
		p.memoryRegions[gpuID].fetchingLayerIndex++
		p.Schedule(playNextTraceEvent{
			time:    p.CurrentTime(),
			handler: p,
			gpuID:   gpuID,
		})
		return
	}

	fmt.Println(p.CurrentTime(), ", gpu, "+strconv.Itoa(gpuID)+
		", fetching start at LayerIndex,"+strconv.Itoa(p.memoryRegions[gpuID].fetchingLayerIndex))
	err := p.msgPkgToSend(p.defaultMemoryRegion.Name, p.memoryRegions[gpuID].Name, tensors, gpuID, "fetch")
	if !err {
		return
	}
}

func (p *DataParallelTracePlayer) nextTensorPkgToMove(
	layer *triosim.Layer,
	gpuID int,
) (tensors []triosim.Tensor, exist bool, reduceTensors []triosim.Tensor, reduce bool) {
	reduce = false
	reduceTensors = make([]triosim.Tensor, 0)
	for _, tensor := range layer.Inputs {
		if !p.isTensorReady(tensor, gpuID) {
			tensors = append(tensors, tensor)
		}
		if layer.Stage == "backward" && p.isTensorInGradientSet(tensor) {
			reduceTensors = append(reduceTensors, tensor)
			reduce = true
		}
	}

	if tensors != nil {
		return tensors, true, reduceTensors, reduce
	}
	if layer.Name == "aten::_foreach_addcdiv_" {
		// if layer.Name == "aten::_foreach_addcdiv_" || layer.ID == p.trace[len(p.trace)-1].ID {
		leftgradtensors := make([]triosim.Tensor, 0)
		for tensor, used := range p.gradientSet {
			if !used {
				leftgradtensors = append(leftgradtensors, tensor)
			}
		}
		reduceTensors = append(reduceTensors, leftgradtensors...)
		fmt.Println("reduce tensors", reduceTensors)
	}

	return nil, false, reduceTensors, reduce
}

func (p *DataParallelTracePlayer) isTensorInGradientSet(tensor triosim.Tensor) bool {
	_, exists := p.gradientSet[tensor]
	return exists
}

func (p *DataParallelTracePlayer) isTensorReady(
	tensor triosim.Tensor,
	gpuID int,
) bool {
	region := p.memoryRegions[gpuID]

	_, ok := region.Tensors[tensor.ID]

	return ok
}

func (p *DataParallelTracePlayer) allTensorsOfLayerAreAvailable(
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

func (p *DataParallelTracePlayer) checkSpaceForTensors(
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

	if purpose == "reduce" || purpose == "scatter" || purpose == "gather" {
		reduceLayerIndex := p.reducelayer
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

func (p *DataParallelTracePlayer) addTensorsToMemRegion(
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

func (p *DataParallelTracePlayer) filterTensors(
	tensors []triosim.Tensor,
	gpuID int,
	purpose string,
) []triosim.Tensor {
	region := p.memoryRegions[gpuID]
	var tensorsCheck []triosim.Tensor

	switch purpose {
	case "scatter", "gather", "reduce":
		tensorsCheck = p.handleScatterGatherReduce(tensors, region, purpose)
	case "fetch":
		tensorsCheck = p.handleFetch(tensors, region)
	default: //"allocate", "complete"
		tensorsCheck = p.handleAllocateComplete(tensors, region, purpose)
	}

	return tensorsCheck
}

func (p *DataParallelTracePlayer) handleScatterGatherReduce(
	tensors []triosim.Tensor,
	region *DataMemoryRegion,
	purpose string,
) []triosim.Tensor {
	tensorsCheck := make([]triosim.Tensor, 0)
	for _, tensor := range tensors {
		tensor.ID = tensor.ID + "chunk" + strconv.Itoa(tensor.ChunkID)
		dstTensor, found := region.Tensors[tensor.ID]
		if found {
			if purpose == "scatter" {
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

func (p *DataParallelTracePlayer) handleFetch(
	tensors []triosim.Tensor,
	region *DataMemoryRegion,
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

func (p *DataParallelTracePlayer) handleAllocateComplete(
	tensors []triosim.Tensor,
	region *DataMemoryRegion,
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

func (p *DataParallelTracePlayer) removeTensorFromMemRegion(
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

func (p *DataParallelTracePlayer) doAllReduce() {
	if p.ncclAction != "nextlayer" {
		return
	}
	if p.reducelayer >= len(p.reducetrace) {
		return
	}
	// data chunking
	tensors := p.reducetrace[p.reducelayer].Inputs
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

func (p *DataParallelTracePlayer) doScatter() {
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

func (p *DataParallelTracePlayer) doAllgather() {
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
