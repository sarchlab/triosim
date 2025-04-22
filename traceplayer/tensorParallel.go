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

// A playTPNextReduceEvent triggers the player to continue to play the trace when in Ring AllReduce process.
type playTPNextReduceEvent struct {
	time    sim.VTimeInSec
	handler *TensorParallelTracePlayer
}

// Time returns the time of the event.
func (e playTPNextReduceEvent) Time() sim.VTimeInSec {
	return e.time
}

// Handler returns the handler of the event.
func (e playTPNextReduceEvent) Handler() sim.Handler {
	return e.handler
}

// IsSecondary always returns false.
func (e playTPNextReduceEvent) IsSecondary() bool {
	return false
}

// A playNextTPEvent triggers the player to continue to play the trace.
type playNextTPEvent struct {
	time    sim.VTimeInSec
	handler *TensorParallelTracePlayer
	gpuID   int
}

// GetGPUID returns the GPUID of the event.
func (e playNextTPEvent) GetGPUID() int {
	return e.gpuID
}

// Time returns the time of the event.
func (e playNextTPEvent) Time() sim.VTimeInSec {
	return e.time
}

// Handler returns the handler of the event.
func (e playNextTPEvent) Handler() sim.Handler {
	return e.handler
}

// IsSecondary always returns false.
func (e playNextTPEvent) IsSecondary() bool {
	return false
}

// A tpLayerCompletionEvent is triggered when a layer is completed.
type tpLayerCompletionEvent struct {
	time    sim.VTimeInSec
	handler *TensorParallelTracePlayer
	layer   *triosim.Layer
	gpuID   int
}

// GetGPUID returns the GPUID of the event.
func (e tpLayerCompletionEvent) GetGPUID() int {
	return e.gpuID
}

// Time returns the time of the event.
func (e tpLayerCompletionEvent) Time() sim.VTimeInSec {
	return e.time
}

// Handler returns the handler of the event.
func (e tpLayerCompletionEvent) Handler() sim.Handler {
	return e.handler
}

// IsSecondary always returns false.
func (e tpLayerCompletionEvent) IsSecondary() bool {
	return false
}

// TensorParallelMemoryRegion describes a region of memory that can hold tensors.
type TensorParallelMemoryRegion struct {
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
func (r *TensorParallelMemoryRegion) TotalUtilizedBytes() uint64 {
	var total uint64

	for _, t := range r.Tensors {
		total += t.Bytes()
	}

	return total
}

// A TensorParallelTracePlayer replays the forward path of a trace.
type TensorParallelTracePlayer struct {
	*sim.ComponentBase

	sim.TimeTeller
	sim.EventScheduler
	timeEstimator timemodel.TimeEstimator

	memoryRegions       []*TensorParallelMemoryRegion
	defaultMemoryRegion *TensorParallelMemoryRegion

	trace        triosim.Trace
	batchSize    int
	reducelayer  int
	ncclAction   string
	sendTofinish int
	gpuchunks    [][]triosim.Tensor
	scatterstep  int
	gatherstep   int
}

// NewTensorParallelTracePlayer creates a new TensorParallelTracePlayer.
func NewTensorParallelTracePlayer(
	name string,
	tt sim.TimeTeller,
	es sim.EventScheduler,
	timeEstimator timemodel.TimeEstimator,
) *TensorParallelTracePlayer {
	p := &TensorParallelTracePlayer{
		timeEstimator:  timeEstimator,
		TimeTeller:     tt,
		EventScheduler: es,
	}

	p.ComponentBase = sim.NewComponentBase(name)

	return p
}

// AddMemoryRegion adds a memory region to the player.
func (p *TensorParallelTracePlayer) AddMemoryRegion(
	region *TensorParallelMemoryRegion,
	port sim.Port,
) {
	p.memoryRegions = append(p.memoryRegions, region)
	p.AddPort(region.Name, port)
}

// SetDefaultMemoryRegion sets the default memory region.
func (p *TensorParallelTracePlayer) SetDefaultMemoryRegion(region *TensorParallelMemoryRegion) {
	p.defaultMemoryRegion = region
}

// Handle function of a TensorParallelTracePlayer handles events.
func (p *TensorParallelTracePlayer) Handle(e sim.Event) error {
	switch e := e.(type) {
	case playNextTPEvent:
		gpuID := e.gpuID
		p.playNext(gpuID)
	case tpLayerCompletionEvent:
		p.completeLayer(e)
	case playTPNextReduceEvent:
		p.playNextReduce()
	default:
		panic("TensorParallelTracePlayer cannot handle this event type " +
			reflect.TypeOf(e).String())
	}

	return nil
}

// NotifyPortFree function of a TensorParallelTracePlayer notifies that the
// one port of the component if free.
func (p *TensorParallelTracePlayer) NotifyPortFree(
	now sim.VTimeInSec,
	port sim.Port,
) {
	msg := port.Retrieve(now)
	msginfo := msg.(*triosim.TensorMsg)
	if msginfo.Purpose == "scatter" || msginfo.Purpose == "gather" {
		p.playNextReduce()
		if p.ncclAction == "nextlayer" {
			for _, item := range p.memoryRegions {
				p.playNext(item.GPUID)
			}
		}
	} else {
		p.playNext(msginfo.GPUID)
	}
}

// NotifyRecv function notifies that the component has received a message.
func (p *TensorParallelTracePlayer) NotifyRecv(
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
			// fmt.Println(p.CurrentTime(), ", gpu "+strconv.Itoa(gpuID), ", ", msg.Purpose)
			p.Schedule(playTPNextReduceEvent{
				time:    p.CurrentTime(),
				handler: p,
			})

			if p.ncclAction == "nextlayer" {
				for _, item := range p.memoryRegions {
					p.Schedule(playNextTPEvent{
						time:    p.CurrentTime(),
						handler: p,
						gpuID:   item.GPUID,
					})
				}
			}
		} else {
			p.Schedule(playNextTPEvent{
				time:    p.CurrentTime(),
				handler: p,
				gpuID:   gpuID,
			})
		}
	default:
		panic(fmt.Sprintf("Cannot handle message %T", msg))
	}
}

func (p *TensorParallelTracePlayer) recvTensorPkg(msg *triosim.TensorMsg) {
	p.removeInflightTransfer(msg)
	p.addTensorsToMemRegion(msg.TensorPkg, msg.GPUID, msg.Purpose)
	if msg.Purpose == "scatter" || msg.Purpose == "gather" {
		p.sendTofinish--
	}
}

func (p *TensorParallelTracePlayer) removeInflightTransfer(msg *triosim.TensorMsg) {
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

// SetTrace sets the trace to replay by the TensorParallelTracePlayer.
func (p *TensorParallelTracePlayer) SetTrace(
	trace triosim.Trace,
	batchSize int,
) {
	if p.defaultMemoryRegion == nil {
		panic("DefaultTensorParallelMemoryRegion is not set")
	}

	p.addTensorsToDefaultMemRegion(trace)

	p.trace = trace
	p.batchSize = batchSize
}

// TensorParallelStart starts the simulation. It will schedule the first playNextTPEvent.
// The main program should still call engine.run() to run the simulation.
func (p *TensorParallelTracePlayer) TensorParallelStart() {
	if (p.trace == nil) || (len(p.trace) == 0) {
		panic("Trace is not set")
	}

	for _, item := range p.memoryRegions {
		fmt.Printf("Memory region %s \n", item.Name)
		p.Schedule(playNextTPEvent{
			time:    p.CurrentTime(),
			handler: p,
			gpuID:   item.GPUID,
		})
	}
}

func (p *TensorParallelTracePlayer) addTensorsToDefaultMemRegion(
	trace triosim.Trace,
) {
	for _, layer := range trace {
		for _, tensor := range layer.Inputs {
			p.defaultMemoryRegion.Tensors[tensor.ID] = tensor
		}
	}
}

func (p *TensorParallelTracePlayer) msgPkgToSend(
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
	// fmt.Println(p.CurrentTime()*1000000, ", gpu "+strconv.Itoa(gpuID)+
	//" send "+purpose+" "+strconv.Itoa(totalBytes)+" bytes")
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
func (p *TensorParallelTracePlayer) playNext(gpuID int) {
	p.doFetching(gpuID)
	p.doComputing(gpuID)
}

// playNextReduce performs the next action that replays the trace when in Ring Allreduce process
func (p *TensorParallelTracePlayer) playNextReduce() {
	p.doScatter()
	p.doAllgather()
}

func (p *TensorParallelTracePlayer) divideTensorToMicroBatch(
	layer *triosim.Layer,
) {
	if !layer.SetBatchSize {
		if layer.TPflag == 1 {
			numMicroBatch := float64(len(p.memoryRegions) - 1)
			for i := range layer.Inputs {
				layer.Inputs[i].Size = int(float64(layer.Inputs[i].Size) / numMicroBatch)
				layer.InputSize[i] = int(float64(layer.InputSize[i]) / numMicroBatch)
			}
			for i := range layer.Outputs {
				layer.Outputs[i].Size = int(float64(layer.Outputs[i].Size) / numMicroBatch)
				layer.OutputSize[i] = int(float64(layer.OutputSize[i]) / numMicroBatch)
			}

			layer.TimeInSec = layer.TimeInSec / numMicroBatch
		}

		layer.SetBatchSize = true
	}
}

func (p *TensorParallelTracePlayer) doFetching(gpuID int) {
	if len(p.memoryRegions[gpuID].inflightTransfer) > 0 {
		return
	}

	if p.memoryRegions[gpuID].fetchingLayerIndex >= len(p.trace) {
		return
	}

	if p.memoryRegions[gpuID].fetchingLayerIndex < p.memoryRegions[gpuID].computingLayerIndex {
		p.memoryRegions[gpuID].fetchingLayerIndex++
		p.Schedule(playNextTPEvent{
			time:    p.CurrentTime(),
			handler: p,
			gpuID:   gpuID,
		})
		return
	}

	p.logStepDone(gpuID, "fetching check", p.memoryRegions[gpuID].fetchingLayerIndex)
	if p.allTensorsOfLayerAreAvailable(p.trace[p.memoryRegions[gpuID].fetchingLayerIndex], gpuID) {
		p.memoryRegions[gpuID].fetchingLayerIndex++
		p.doFetching(gpuID)
		return
	}

	p.divideTensorToMicroBatch(p.trace[p.memoryRegions[gpuID].fetchingLayerIndex])
	tensors, needsMoving := p.nextTensorPkgToMove(
		p.trace[p.memoryRegions[gpuID].fetchingLayerIndex], gpuID)
	if !p.addTensorsToMemRegion(tensors, gpuID, "allocate") {
		return
	}

	p.addTensorsToMemRegion(p.trace[p.memoryRegions[gpuID].fetchingLayerIndex].Outputs, gpuID, "allocate")
	if !needsMoving {
		p.memoryRegions[gpuID].fetchingLayerIndex++
		p.Schedule(playNextTPEvent{
			time:    p.CurrentTime(),
			handler: p,
			gpuID:   gpuID,
		})
		return
	}
	p.logStepDone(gpuID, "fetching start", p.memoryRegions[gpuID].fetchingLayerIndex)
	err := p.msgPkgToSend(p.defaultMemoryRegion.Name, p.memoryRegions[gpuID].Name, tensors, gpuID, "fetch")
	if !err {
		return
	}
}

func (p *TensorParallelTracePlayer) doComputing(gpuID int) {
	if p.reducelayer < p.memoryRegions[gpuID].computingLayerIndex {
		return
	}

	if p.memoryRegions[gpuID].doingComputing {
		return
	}

	if p.memoryRegions[gpuID].computingLayerIndex >= len(p.trace) {
		return
	}

	layer := p.trace[p.memoryRegions[gpuID].computingLayerIndex]
	p.logStepDone(gpuID, "computing check", p.memoryRegions[gpuID].computingLayerIndex)
	p.divideTensorToMicroBatch(p.trace[p.memoryRegions[gpuID].computingLayerIndex])
	_, needsMoving := p.nextTensorPkgToMove(layer, gpuID)
	if needsMoving {
		return
	}

	p.logStepDone(gpuID, "computing start", p.memoryRegions[gpuID].computingLayerIndex)
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
	evt := tpLayerCompletionEvent{
		time:    now + sim.VTimeInSec(output.TimeInSec),
		handler: p,
		layer:   layer,
		gpuID:   gpuID,
	}
	p.Schedule(evt)

	p.memoryRegions[gpuID].doingComputing = true
	p.memoryRegions[gpuID].computingLayerIndex++
}

func (p *TensorParallelTracePlayer) completeLayer(e sim.Event) {
	evt := e.(tpLayerCompletionEvent)
	layer := evt.layer
	gpuID := evt.gpuID

	p.addTensorsToMemRegion(layer.Outputs, gpuID, "complete")
	p.memoryRegions[gpuID].doingComputing = false
	p.logStepDone(gpuID, "computing done", p.memoryRegions[gpuID].computingLayerIndex-1)
	p.reducelayer = p.memoryRegions[gpuID].computingLayerIndex - 1
	layername := p.trace[p.memoryRegions[gpuID].computingLayerIndex-1].Name
	if p.trace[p.memoryRegions[gpuID].computingLayerIndex-1].TPflag == 1 {
		fmt.Println("-----------", layername, p.reducelayer, gpuID)
		allreduceflag := true
		for i := 0; i < len(p.memoryRegions)-1; i++ {
			if p.reducelayer != p.memoryRegions[i].computingLayerIndex-1 {
				fmt.Println("computingLayerIndex-1", i, p.memoryRegions[i].computingLayerIndex-1)
				allreduceflag = false
			}
		}
		if allreduceflag {
			if !p.doAllReduce() {
				for _, item := range p.memoryRegions {
					p.Schedule(playNextTPEvent{
						time:    p.CurrentTime(),
						handler: p,
						gpuID:   item.GPUID,
					})
				}
			}
		}
	} else {
		p.reducelayer++
		p.Schedule(playNextTPEvent{
			time:    p.CurrentTime(),
			handler: p,
			gpuID:   gpuID,
		})
	}
}

func (p *TensorParallelTracePlayer) doAllReduce() bool {
	p.ncclAction = "scatter"
	if p.reducelayer >= len(p.trace) {
		return false
	}
	if len(p.memoryRegions) <= 2 {
		return false
	}
	// data chunking
	tensors := p.trace[p.reducelayer].Outputs
	if tensors == nil {
		p.Schedule(playTPNextReduceEvent{
			time:    p.CurrentTime(),
			handler: p,
		})
		return true
	}
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
	p.Schedule(playTPNextReduceEvent{
		time:    p.CurrentTime(),
		handler: p,
	})
	return true
}

func (p *TensorParallelTracePlayer) doScatter() {
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

func (p *TensorParallelTracePlayer) doAllgather() {
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

func (p *TensorParallelTracePlayer) nextTensorPkgToMove(
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

func (p *TensorParallelTracePlayer) isTensorReady(
	tensor triosim.Tensor,
	gpuID int,
) bool {
	region := p.memoryRegions[gpuID]

	_, ok := region.Tensors[tensor.ID]

	return ok
}

func (p *TensorParallelTracePlayer) allTensorsOfLayerAreAvailable(
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

func (p *TensorParallelTracePlayer) checkSpaceForTensors(
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
		p.removeTensorFromMemRegion(gpuID, true)
	} else {
		p.removeTensorFromMemRegion(gpuID, false)
	}

	totalBytes = region.TotalUtilizedBytes() + totalTensorBytes
	if totalBytes <= region.Capacity {
		return true
	}

	fmt.Println("region is full after deleting")
	return false
}

func (p *TensorParallelTracePlayer) addTensorsToMemRegion(
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
		default: //reduce, gather, scatter
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

func (p *TensorParallelTracePlayer) filterTensors(
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

func (p *TensorParallelTracePlayer) handleScatterGatherReduce(
	tensors []triosim.Tensor,
	region *TensorParallelMemoryRegion,
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

func (p *TensorParallelTracePlayer) handleFetch(
	tensors []triosim.Tensor,
	region *TensorParallelMemoryRegion,
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

func (p *TensorParallelTracePlayer) handleAllocateComplete(
	tensors []triosim.Tensor,
	region *TensorParallelMemoryRegion,
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

func (p *TensorParallelTracePlayer) removeTensorFromMemRegion(
	gpuID int,
	forreduce bool,
) {
	region := p.memoryRegions[gpuID]
	removed := false
	existingTensorIDs := make(map[string]struct{})
	var layer *triosim.Layer
	var LayerTensors []triosim.Tensor
	if forreduce {
		layer = p.trace[p.reducelayer]
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

func (p *TensorParallelTracePlayer) logStepDone(gpuID int, currentStep string, layerIndex int) {
	fmt.Println(p.CurrentTime(), ", gpu "+strconv.Itoa(gpuID)+
		", "+currentStep+" at LayerIndex,"+strconv.Itoa(layerIndex))
}
