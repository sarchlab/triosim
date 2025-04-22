package main

import (
	"flag"
	"fmt"
	"net/http"
	_ "net/http/pprof"
	"strconv"
	"time"

	"github.com/syifan/triosim"
	"github.com/syifan/triosim/networkmodel"
	"github.com/syifan/triosim/timemodel"
	"github.com/syifan/triosim/traceplayer"
	"gitlab.com/akita/akita/v3/sim"
)

var traceDir = flag.String("trace-dir", "../sample_trace/trace2-h100-bs128/vgg13/",
	"The directory where the trace files are located.")
var batchSizeTrace = flag.Int("batch-size", 128, "The batch size of the original trace.")
var batchSizeSim = flag.Int("batch-size-sim", -1, "The simulation batch size, defaults to batchSizeTrace.")
var bandwidth = flag.Float64("bandwidth", 696, "The bandwidth of the link between each GPU and remote memory in GBps.")
var ptpbandwidth = flag.Float64("ptp-bandwidth", 65, "The bandwidth of the link among GPUs in GBps.")
var GPUNumber = flag.Int("GPUnumber", 8, "The number of the GPUs.")
var MicroBatchSize = flag.Int("micro-batch-size", -1, "The micro batch size in the pipeline parallelism.")
var Case = flag.Int("case", 0, "0: training, 1: standard data parallel, "+
	"2: distributed data parallel, 3: tensor parallel, 4: pipeline parallel")
var capacity = flag.Int("capacity", 40, "The memory capacity of each device, (1 << capacity).")
var numCols = flag.Int("numCols", -1, "The column of mesh  in optical network.")
var numRows = flag.Int("numRows", 1, "The row of mesh  in optical network.")
var interconnects = flag.Int("interconnects", 0, "0: electrical (default) or 1: optical")

// var backupNum = flag.Int("backupNum", 0, "The number of backup workers.")

func main() {
	flag.Parse()
	if *batchSizeSim == -1 {
		*batchSizeSim = *batchSizeTrace
	}
	if *MicroBatchSize == -1 {
		*MicroBatchSize = *batchSizeSim
	}
	if *numCols == -1 {
		*numCols = *GPUNumber
	}
	// Server for pprof
	go func() {
		fmt.Println(http.ListenAndServe("localhost:6060", nil))
	}()

	trace := loadTrace(*batchSizeTrace, *batchSizeSim)
	engine := sim.NewSerialEngine()
	// timeEstimator := &timemodel.AlwaysOneTimeEstimator{}
	timeEstimator := &timemodel.RecordedTimeEstimator{}
	start := time.Now()
	switch *Case {
	case 1:
		playTrace(trace, engine, timeEstimator)
		playTraceWithAllReduce(trace, engine, timeEstimator)
	case 2:
		playDataTrace(trace, engine, timeEstimator)
	case 3:
		playTensorTrace(trace, engine, timeEstimator)
	case 4:
		playPipeTrace(trace, engine, timeEstimator)
	case 5:
		playTraceWithHop(trace, engine, timeEstimator)
	default:
		playTrace(trace, engine, timeEstimator)
	}
	elapsed := time.Since(start)
	fmt.Printf("Program Execution time: %s\n", elapsed)
}

func playTrace(trace triosim.Trace, engine *sim.SerialEngine, timeEstimator timemodel.TimeEstimator) {
	tracePlayer := traceplayer.NewInferenceTracePlayer(
		"Player",
		engine,
		engine,
		timeEstimator,
	)
	remoteMemRegion, remotePort := buildHardwarePlatform(tracePlayer, engine)

	tracePlayer.AddMemoryRegion(remoteMemRegion, remotePort)
	tracePlayer.SetDefaultMemoryRegion(remoteMemRegion)
	tracePlayer.SetTrace(trace, *batchSizeSim)

	tracePlayer.KickStart()
	err := engine.Run()
	if err != nil {
		panic(err)
	}
	// fmt.Printf("%s", *traceDir)
	fmt.Printf("Estimated execution time ms, %.10f\n", engine.CurrentTime()*1000)
}

func playTraceWithAllReduce(trace triosim.Trace, engine *sim.SerialEngine, timeEstimator timemodel.TimeEstimator) {
	tracePlayer := traceplayer.NewInferenceTracePlayer(
		"Player",
		engine,
		engine,
		timeEstimator,
	)
	remoteMemRegion, remotePort := buildHardwarePlatform(tracePlayer, engine)
	tracePlayer.AddMemoryRegion(remoteMemRegion, remotePort)
	tracePlayer.SetDefaultMemoryRegion(remoteMemRegion)
	tracePlayer.SetTrace(trace, *batchSizeSim)

	tracePlayer.AllReduceStart()
	err := engine.Run()
	if err != nil {
		panic(err)
	}
	fmt.Printf("Current time after AllReduce stage ms, %.10f\n", engine.CurrentTime()*1000)
}

func playDataTrace(trace triosim.Trace, engine *sim.SerialEngine, timeEstimator timemodel.TimeEstimator) {
	tracePlayer := traceplayer.NewDataParallelTracePlayer(
		"Player",
		engine,
		engine,
		timeEstimator,
	)
	remoteMemRegion, remotePort := buildDataHardwarePlatform(tracePlayer, engine)
	tracePlayer.AddMemoryRegion(remoteMemRegion, remotePort)
	tracePlayer.SetDefaultMemoryRegion(remoteMemRegion)
	tracePlayer.SetTrace(trace, *batchSizeSim)

	tracePlayer.KickStart()
	err := engine.Run()
	if err != nil {
		panic(err)
	}
	// fmt.Printf("%s", *traceDir)
	fmt.Printf("Current time after Data parallesim execution ms, %.10f\n", engine.CurrentTime()*1000)
}

func playPipeTrace(trace triosim.Trace, engine *sim.SerialEngine, timeEstimator timemodel.TimeEstimator) {
	pipeTracePlayer := traceplayer.NewPipeParallelTracePlayer(
		"Player",
		engine,
		engine,
		timeEstimator,
	)
	remoteMemRegion, remotePort := buildPipeHardwarePlatform(pipeTracePlayer, engine)
	pipeTracePlayer.AddMemoryRegion(remoteMemRegion, remotePort)
	pipeTracePlayer.SetDefaultMemoryRegion(remoteMemRegion)
	pipeTracePlayer.SetTrace(trace, *batchSizeSim)
	pipeTracePlayer.SetMicroBatch(*MicroBatchSize)
	pipeTracePlayer.PipelineStart()
	err := engine.Run()
	if err != nil {
		panic(err)
	}
	// fmt.Printf("%s", *traceDir)
	fmt.Printf("Current time after Pipeline parallesim execution ms, %.10f\n", engine.CurrentTime()*1000)
}

func playTensorTrace(trace triosim.Trace, engine *sim.SerialEngine, timeEstimator timemodel.TimeEstimator) {
	tensorTracePlayer := traceplayer.NewTensorParallelTracePlayer(
		"Player",
		engine,
		engine,
		timeEstimator,
	)
	remoteMemRegion, remotePort := buildTensorHardwarePlatform(tensorTracePlayer, engine)
	tensorTracePlayer.AddMemoryRegion(remoteMemRegion, remotePort)
	tensorTracePlayer.SetDefaultMemoryRegion(remoteMemRegion)
	tensorTracePlayer.SetTrace(trace, *batchSizeSim)
	tensorTracePlayer.TensorParallelStart()
	err := engine.Run()
	if err != nil {
		panic(err)
	}
	// fmt.Printf("%s\n", *traceDir)
	fmt.Printf("Current time after Tensor parallesim execution ms, %.10f\n", engine.CurrentTime()*1000)
}

func playTraceWithHop(trace triosim.Trace, engine *sim.SerialEngine, timeEstimator timemodel.TimeEstimator) {
	tracePlayer := traceplayer.NewInferenceTracePlayer(
		"Player",
		engine,
		engine,
		timeEstimator,
	)
	remoteMemRegion, remotePort, networkModel := buildHardwarePlatformHop(tracePlayer, engine)
	tracePlayer.AddMemoryRegion(remoteMemRegion, remotePort)
	tracePlayer.SetDefaultMemoryRegion(remoteMemRegion)
	tracePlayer.SetTrace(trace, *batchSizeSim)
	tracePlayer.SetNetworkModel(networkModel)
	// tracePlayer.HopAllReduceStart(*backupNum)
	tracePlayer.HopAllReduceStart(0)
	// backupNum should less than the number of each GPU's neighbors, used for HOP paper test
	err := engine.Run()
	if err != nil {
		panic(err)
	}
	// fmt.Printf("%s", *traceDir)
	fmt.Printf("Current time after HOP test ms,, %.10f\n", engine.CurrentTime()*1000)
}

func loadTrace(batchSizeTrace int, batchSizeSim int) triosim.Trace {
	bsRatio := float64(batchSizeTrace) / float64(batchSizeSim)
	traceLoader := triosim.TraceLoader{
		Dir: *traceDir,
	}

	trace, err := traceLoader.Load(bsRatio)
	if err != nil {
		panic(err)
	}

	return trace
}

func buildHardwarePlatform(
	tracePlayer *traceplayer.InferenceTracePlayer,
	engine *sim.SerialEngine,
) (*traceplayer.MemoryRegion, *sim.LimitNumMsgPort) {
	var gpuID []int
	var gpuPortID []*sim.LimitNumMsgPort
	var gpuMemRegionID []*traceplayer.MemoryRegion

	numGPUID := *GPUNumber
	for id := 0; id < numGPUID; id++ {
		name := "GPU" + strconv.Itoa(id)
		gpuMemRegion := &traceplayer.MemoryRegion{
			Name:            name,
			CapacityLimited: true,
			Capacity:        1 << *capacity,
			GPUID:           id,
			Tensors:         make(map[string]triosim.Tensor),
		}
		gpuPort := sim.NewLimitNumMsgPort(tracePlayer, 1, name+"Port")
		tracePlayer.AddMemoryRegion(gpuMemRegion, gpuPort)
		gpuID = append(gpuID, id)
		gpuPortID = append(gpuPortID, gpuPort)
		gpuMemRegionID = append(gpuMemRegionID, gpuMemRegion)
	}

	remoteMemRegion := &traceplayer.MemoryRegion{
		Name:            "Remote",
		CapacityLimited: false,
		Tensors:         make(map[string]triosim.Tensor),
	}

	remotePort := sim.NewLimitNumMsgPort(tracePlayer, 1, "RemotePort")
	if *interconnects == 1 {
		setupOpticalNetwork(engine, remotePort, gpuPortID, numGPUID)
	} else {
		busbandwidth := *ptpbandwidth * 2 * (float64(numGPUID) - 1) / float64(numGPUID)
		setupPacketSwitchingNetwork(engine, remotePort, gpuPortID, numGPUID, busbandwidth)
	}

	return remoteMemRegion, remotePort
}

func buildDataHardwarePlatform(
	tracePlayer *traceplayer.DataParallelTracePlayer,
	engine *sim.SerialEngine,
) (*traceplayer.DataMemoryRegion, *sim.LimitNumMsgPort) {
	var gpuID []int
	var gpuPortID []*sim.LimitNumMsgPort
	var gpuMemRegionID []*traceplayer.DataMemoryRegion

	numGPUID := *GPUNumber
	for id := 0; id < numGPUID; id++ {
		name := "GPU" + strconv.Itoa(id)
		gpuMemRegion := &traceplayer.DataMemoryRegion{
			Name:            name,
			CapacityLimited: true,
			Capacity:        1 << *capacity,
			GPUID:           id,
			Tensors:         make(map[string]triosim.Tensor),
		}
		gpuPort := sim.NewLimitNumMsgPort(tracePlayer, 1, name+"Port")
		tracePlayer.AddMemoryRegion(gpuMemRegion, gpuPort)
		gpuID = append(gpuID, id)
		gpuPortID = append(gpuPortID, gpuPort)
		gpuMemRegionID = append(gpuMemRegionID, gpuMemRegion)
	}

	remoteMemRegion := &traceplayer.DataMemoryRegion{
		Name:            "Remote",
		CapacityLimited: false,
		Tensors:         make(map[string]triosim.Tensor),
	}

	remotePort := sim.NewLimitNumMsgPort(tracePlayer, 1, "RemotePort")
	if *interconnects == 1 {
		setupOpticalNetwork(engine, remotePort, gpuPortID, numGPUID)
	} else {
		busbandwidth := *ptpbandwidth * 2 * (float64(numGPUID) - 1) / float64(numGPUID)
		setupPacketSwitchingNetwork(engine, remotePort, gpuPortID, numGPUID, busbandwidth)
	}
	return remoteMemRegion, remotePort
}

func buildPipeHardwarePlatform(
	tracePlayer *traceplayer.PipeParallelTracePlayer,
	engine *sim.SerialEngine,
) (*traceplayer.PipeMemoryRegion, *sim.LimitNumMsgPort) {
	var gpuID []int
	var gpuPortID []*sim.LimitNumMsgPort
	var gpuMemRegionID []*traceplayer.PipeMemoryRegion

	numGPUID := *GPUNumber
	for id := 0; id < numGPUID; id++ {
		name := "GPU" + strconv.Itoa(id)
		gpuMemRegion := &traceplayer.PipeMemoryRegion{
			Name:            name,
			CapacityLimited: true,
			Capacity:        1 << *capacity,
			GPUID:           id,
			Tensors:         make(map[string]triosim.Tensor),
		}
		gpuPort := sim.NewLimitNumMsgPort(tracePlayer, 1, name+"Port")
		tracePlayer.AddMemoryRegion(gpuMemRegion, gpuPort)
		gpuID = append(gpuID, id)
		gpuPortID = append(gpuPortID, gpuPort)
		gpuMemRegionID = append(gpuMemRegionID, gpuMemRegion)
	}

	remoteMemRegion := &traceplayer.PipeMemoryRegion{
		Name:            "Remote",
		CapacityLimited: false,
		Tensors:         make(map[string]triosim.Tensor),
	}

	remotePort := sim.NewLimitNumMsgPort(tracePlayer, 1, "RemotePort")

	if *interconnects == 1 {
		setupOpticalNetwork(engine, remotePort, gpuPortID, numGPUID)
	} else {
		setupPacketSwitchingNetwork(engine, remotePort, gpuPortID, numGPUID, *ptpbandwidth)
	}
	return remoteMemRegion, remotePort
}

func buildTensorHardwarePlatform(
	tracePlayer *traceplayer.TensorParallelTracePlayer,
	engine *sim.SerialEngine,
) (*traceplayer.TensorParallelMemoryRegion, *sim.LimitNumMsgPort) {
	var gpuID []int
	var gpuPortID []*sim.LimitNumMsgPort
	var gpuMemRegionID []*traceplayer.TensorParallelMemoryRegion

	numGPUID := *GPUNumber
	for id := 0; id < numGPUID; id++ {
		name := "GPU" + strconv.Itoa(id)
		gpuMemRegion := &traceplayer.TensorParallelMemoryRegion{
			Name:            name,
			CapacityLimited: true,
			Capacity:        1 << *capacity,
			GPUID:           id,
			Tensors:         make(map[string]triosim.Tensor),
		}
		gpuPort := sim.NewLimitNumMsgPort(tracePlayer, 1, name+"Port")
		tracePlayer.AddMemoryRegion(gpuMemRegion, gpuPort)
		gpuID = append(gpuID, id)
		gpuPortID = append(gpuPortID, gpuPort)
		gpuMemRegionID = append(gpuMemRegionID, gpuMemRegion)
	}

	remoteMemRegion := &traceplayer.TensorParallelMemoryRegion{
		Name:            "Remote",
		CapacityLimited: false,
		Tensors:         make(map[string]triosim.Tensor),
	}

	remotePort := sim.NewLimitNumMsgPort(tracePlayer, 1, "RemotePort")
	if *interconnects == 1 {
		setupOpticalNetwork(engine, remotePort, gpuPortID, numGPUID)
	} else {
		busbandwidth := *ptpbandwidth * 2 * (float64(numGPUID) - 1) / float64(numGPUID)
		setupPacketSwitchingNetwork(engine, remotePort, gpuPortID, numGPUID, busbandwidth)
	}
	return remoteMemRegion, remotePort
}

func buildHardwarePlatformHop(
	tracePlayer *traceplayer.InferenceTracePlayer,
	engine *sim.SerialEngine,
) (*traceplayer.MemoryRegion, *sim.LimitNumMsgPort, *networkmodel.PacketSwitchingNetworkModel) {
	var gpuID []int
	var gpuPortID []*sim.LimitNumMsgPort
	var gpuMemRegionID []*traceplayer.MemoryRegion

	numGPUID := *GPUNumber
	for id := 0; id < numGPUID; id++ {
		name := "GPU" + strconv.Itoa(id)
		gpuMemRegion := &traceplayer.MemoryRegion{
			Name:            name,
			CapacityLimited: true,
			Capacity:        1 << *capacity,
			GPUID:           id,
			Tensors:         make(map[string]triosim.Tensor),
		}
		gpuPort := sim.NewLimitNumMsgPort(tracePlayer, 1, name+"Port")
		tracePlayer.AddMemoryRegion(gpuMemRegion, gpuPort)
		gpuID = append(gpuID, id)
		gpuPortID = append(gpuPortID, gpuPort)
		gpuMemRegionID = append(gpuMemRegionID, gpuMemRegion)
	}

	remoteMemRegion := &traceplayer.MemoryRegion{
		Name:            "Remote",
		CapacityLimited: false,
		Tensors:         make(map[string]triosim.Tensor),
	}

	remotePort := sim.NewLimitNumMsgPort(tracePlayer, 1, "RemotePort")
	networkModel := networkmodel.NewPacketSwitchingNetworkModel(engine, engine)
	networkModel.PlugInWithDetails(remotePort, 1, "")
	for i := 0; i < numGPUID; i++ {
		networkModel.PlugInWithDetails(gpuPortID[i], 1, "")
		networkModel.AddLink(gpuPortID[i], remotePort, *bandwidth*1e9, 1e-7)
	}
	//ring based
	for i := 0; i < numGPUID; i++ {
		fmt.Println(gpuPortID[i], gpuPortID[(i+1)%numGPUID])
		fmt.Println(gpuPortID[i], gpuPortID[(i+numGPUID/2)%numGPUID])
		if i < numGPUID/2 {
			networkModel.AddLink(gpuPortID[i], gpuPortID[(i+1)%numGPUID], *ptpbandwidth*1e9, 1e-7)
			networkModel.AddLink(gpuPortID[i], gpuPortID[(i+numGPUID/2)%numGPUID], *ptpbandwidth*1e9, 1e-7)
		} else {
			networkModel.AddLink(gpuPortID[i], gpuPortID[(i+1)%numGPUID], *ptpbandwidth*1e9, 1e-7)
			// networkModel.AddLink(gpuPortID[i], gpuPortID[(i+numGPUID/2)%numGPUID], *ptpbandwidth*1e9, 1e-7)
		}
	}

	return remoteMemRegion, remotePort, networkModel
}

func setupPacketSwitchingNetwork(
	engine *sim.SerialEngine,
	remotePort *sim.LimitNumMsgPort,
	gpuPortID []*sim.LimitNumMsgPort,
	numGPUID int,
	busbandwidth float64,
) {
	networkModel := networkmodel.NewPacketSwitchingNetworkModel(engine, engine)
	networkModel.PlugInWithDetails(remotePort, 1, "")
	for i := 0; i < numGPUID; i++ {
		networkModel.PlugInWithDetails(gpuPortID[i], 1, "")
		networkModel.AddLink(gpuPortID[i], remotePort, *bandwidth*1e9, 1e-7)
	}
	// ring-based all reduce
	for i := 0; i < numGPUID; i++ {
		networkModel.AddLink(gpuPortID[i], gpuPortID[(i+1)%numGPUID], busbandwidth*1e9, 1e-7)
	}
}

func setupOpticalNetwork(
	engine *sim.SerialEngine,
	remotePort *sim.LimitNumMsgPort,
	gpuPortID []*sim.LimitNumMsgPort,
	numGPUID int,
) {
	networkModel := networkmodel.NewOpticalNetworkModel(engine, engine, 8, 0.02)
	networkModel.PlugIn(remotePort, 1)
	for i := 0; i < numGPUID; i++ {
		networkModel.PlugIn(gpuPortID[i], 1)
	}
	if (*numRows)*(*numCols) != numGPUID {
		panic("numrow*numcol != numGPUID")
	}
	networkModel.InitHardwareNetwork("mesh", *numRows, *numCols)
	networkModel.InitLogicalNetwork("ring", *numRows, *numCols)
	// networkModel.InitLogicalNetwork("butterfly", numRows, numCols)
	//destroy the network
	//networkModel.Destroy()
	//create the new network
}
