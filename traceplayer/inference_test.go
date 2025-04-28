package traceplayer

import (
	gomock "github.com/golang/mock/gomock"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	"github.com/sarchlab/triosim"
	"github.com/sarchlab/triosim/timemodel"
	"gitlab.com/akita/akita/v3/sim"
)

var _ = Describe("Inference Player", func() {
	var (
		mockCtrl        *gomock.Controller
		tt              *MockTimeTeller
		es              *MockEventScheduler
		te              *MockTimeEstimator
		player          *InferenceTracePlayer
		localMemRegion  *MemoryRegion
		GPUMemRegion    *MemoryRegion
		remoteMemRegion *MemoryRegion
		localPort       *MockPort
		remotePort      *MockPort
		GPUPort         *MockPort
		trace           triosim.Trace
	)

	BeforeEach(func() {
		mockCtrl = gomock.NewController(GinkgoT())
		tt = NewMockTimeTeller(mockCtrl)
		es = NewMockEventScheduler(mockCtrl)
		te = NewMockTimeEstimator(mockCtrl)
		localPort = NewMockPort(mockCtrl)
		localPort.EXPECT().Name().Return("LocalPort").AnyTimes()
		remotePort = NewMockPort(mockCtrl)
		remotePort.EXPECT().Name().Return("RemotePort").AnyTimes()
		GPUPort = NewMockPort(mockCtrl)
		GPUPort.EXPECT().Name().Return("GPU0Port").AnyTimes()

		player = NewInferenceTracePlayer("Player", tt, es, te)

		localMemRegion = &MemoryRegion{
			Name:            "Local",
			CapacityLimited: true,
			Capacity:        1 << 32,
			Tensors:         make(map[string]triosim.Tensor),
			GPUID:           1,
		}

		GPUMemRegion = &MemoryRegion{
			Name:            "GPU0",
			CapacityLimited: true,
			Capacity:        1 << 32,
			Tensors:         make(map[string]triosim.Tensor),
			GPUID:           0,
		}

		remoteMemRegion = &MemoryRegion{
			Name:            "Remote",
			CapacityLimited: false,
			Tensors:         make(map[string]triosim.Tensor),
		}

		player.AddMemoryRegion(localMemRegion, localPort)
		player.AddMemoryRegion(GPUMemRegion, GPUPort)
		player.AddMemoryRegion(remoteMemRegion, remotePort)

		trace = []*triosim.Layer{
			{
				ID:   1,
				Name: "aten::_foreach_addcdiv_",
				Inputs: []triosim.Tensor{
					{ID: "1", Size: 4096},
					{ID: "2", Size: 4096},
					{ID: "3", Size: 4096},
				},
				Outputs: []triosim.Tensor{
					{ID: "4", Size: 4096},
					{ID: "5", Size: 4096, Category: triosim.Gradient},
				},
				InputSize:  []int{12288},
				OutputSize: []int{8192},
			},
		}

	})

	AfterEach(func() {
		mockCtrl.Finish()
	})

	Context("when only using local memory", func() {
		BeforeEach(func() {
			player.SetDefaultMemoryRegion(localMemRegion)
			player.SetTrace(trace, 128)
		})

		It("should do compute", func() {
			tt.EXPECT().CurrentTime().Return(sim.VTimeInSec(0.0)).AnyTimes()
			te.EXPECT().Estimate(gomock.Any()).
				Return(timemodel.TimeEstimatorOutput{
					TimeInSec: 0.1,
				}, nil)
			es.EXPECT().Schedule(layerCompletionEvent{
				time:    0.1,
				handler: player,
				layer:   trace[0],
			})

			player.playNext(0)

			Expect(player.memoryRegions[0].doingComputing).To(BeTrue())
			Expect(player.memoryRegions[0].computingLayerIndex).To(Equal(1))
		})

		It("should finish computing the layer", func() {
			player.memoryRegions[0].doingComputing = true
			player.memoryRegions[0].computingLayerIndex = 1
			tt.EXPECT().CurrentTime().Return(sim.VTimeInSec(0.1)).AnyTimes()
			es.EXPECT().Schedule(playNextEvent{
				time:    0.1,
				handler: player,
			})

			evt := layerCompletionEvent{
				time:    0.1,
				handler: player,
				layer:   trace[0],
				gpuID:   0,
			}
			player.Handle(evt)

			Expect(localMemRegion.Tensors).To(HaveLen(5))
			Expect(localMemRegion.Tensors["4"].ID).To(HaveLen(1))
			Expect(player.memoryRegions[0].doingComputing).To(BeFalse())
		})
	})

	Context("when data is in remote memory", func() {
		BeforeEach(func() {
			player.SetDefaultMemoryRegion(remoteMemRegion)
			player.SetTrace(trace, 128)
			player.gradientSet = make(map[triosim.Tensor]bool)
			for _, layer := range trace {
				for _, tensor := range layer.Inputs {
					player.gradientSet[tensor] = true
				}
			}
		})

		It("should not move data if there is on-going transfer from/to the region", func() {
			tt.EXPECT().CurrentTime().Return(sim.VTimeInSec(0.0)).AnyTimes()
			player.memoryRegions[0].fetchingLayerIndex = 0
			player.memoryRegions[0].computingLayerIndex = 0
			player.memoryRegions[0].inflightTransfer = []*triosim.TensorMsg{
				{
					MsgMeta: sim.MsgMeta{
						Src: remotePort,
						Dst: localPort,
					},
				},
			}

			player.playNext(0)
		})

		It("should move input tensor package", func() {
			tt.EXPECT().CurrentTime().Return(sim.VTimeInSec(0.0)).AnyTimes()
			te.EXPECT().Estimate(gomock.Any()).
				Return(timemodel.TimeEstimatorOutput{
					TimeInSec: 0.1,
				}, nil)
			es.EXPECT().Schedule(layerCompletionEvent{
				time:    0.1,
				handler: player,
				layer:   trace[0],
			})
			remotePort.EXPECT().
				Send(gomock.Any()).
				Do(func(msg *triosim.TensorMsg) {
					Expect(msg.TensorPkg[0].ID).To(Equal("1"))
					Expect(msg.TensorPkg[1].ID).To(Equal("2"))
					Expect(msg.TensorPkg[2].ID).To(Equal("3"))
					Expect(msg.DstRegionName).To(Equal(localMemRegion.Name))
					Expect(msg.Dst).To(Equal(localPort))
					Expect(msg.Src).To(Equal(remotePort))
					Expect(msg.SendTime).To(Equal(sim.VTimeInSec(0.0)))
					Expect(msg.TrafficBytes).
						To(Equal(4096 * 3))
				})

			player.playNext(0)
		})

		It("should add tensor to memory region after move", func() {
			tensorMsg := &triosim.TensorMsg{
				TensorPkg:     []triosim.Tensor{trace[0].Inputs[0], trace[0].Inputs[1]},
				DstRegionName: localMemRegion.Name,
			}
			player.memoryRegions[0].inflightTransfer = []*triosim.TensorMsg{tensorMsg}

			tt.EXPECT().CurrentTime().Return(sim.VTimeInSec(1.1)).AnyTimes()
			localPort.EXPECT().
				Retrieve(sim.VTimeInSec(1.1)).
				Return(tensorMsg)
			es.EXPECT().Schedule(playNextEvent{
				time:    1.1,
				handler: player,
				gpuID:   0,
			})

			player.NotifyRecv(1.1, localPort)

			Expect(player.memoryRegions[0].inflightTransfer).To(HaveLen(0))
			Expect(localMemRegion.Tensors).To(HaveLen(2))
		})

		It("should do all reduce scatter", func() {
			tt.EXPECT().CurrentTime().Return(sim.VTimeInSec(0.0)).AnyTimes()
			player.ncclAction = "nextlayer"
			localPort.EXPECT().
				Send(gomock.Any()).
				Do(func(msg *triosim.TensorMsg) {
					Expect(msg.TensorPkg[0].ID).To(Equal("1"))
					Expect(msg.DstRegionName).To(Equal(GPUMemRegion.Name))
					Expect(msg.GPUID).To(Equal(1))
					Expect(msg.Purpose).To(Equal("scatter"))
					Expect(msg.Dst).To(Equal(GPUPort))
					Expect(msg.Src).To(Equal(localPort))
					Expect(msg.SendTime).To(Equal(sim.VTimeInSec(0.0)))
					Expect(msg.TrafficBytes).
						To(Equal(6144))
				})

			GPUPort.EXPECT().
				Send(gomock.Any()).
				Do(func(msg *triosim.TensorMsg) {
					Expect(msg.TensorPkg[0].ID).To(Equal("1"))
					Expect(msg.DstRegionName).To(Equal(localMemRegion.Name))
					Expect(msg.GPUID).To(Equal(0))
					Expect(msg.Purpose).To(Equal("scatter"))
					Expect(msg.Dst).To(Equal(localPort))
					Expect(msg.Src).To(Equal(GPUPort))
					Expect(msg.TrafficBytes).
						To(Equal(6144))
				})
			player.playNextReduce()
		})

		It("should do all reduce gather", func() {
			gpuChunks := make([][]triosim.Tensor, 0)
			chunk1 := []triosim.Tensor{
				{
					ID:       "5",
					Size:     2048,
					Category: triosim.Gradient,
					ChunkID:  0,
				},
				{
					ID:       "5",
					Size:     2048,
					Category: triosim.Gradient,
					ChunkID:  1,
				},
			}

			chunk2 := []triosim.Tensor{
				{
					ID:       "5",
					Size:     2048,
					Category: triosim.Gradient,
					ChunkID:  0,
				},
				{
					ID:       "5",
					Size:     2048,
					Category: triosim.Gradient,
					ChunkID:  1,
				},
			}

			tt.EXPECT().CurrentTime().Return(sim.VTimeInSec(0.0)).AnyTimes()
			localPort.EXPECT().
				Send(gomock.Any()).
				Do(func(msg *triosim.TensorMsg) {
					Expect(msg.TensorPkg[0].ID).To(Equal("5"))
					Expect(msg.DstRegionName).To(Equal(GPUMemRegion.Name))
					Expect(msg.GPUID).To(Equal(1))
					Expect(msg.Purpose).To(Equal("gather"))
					Expect(msg.Dst).To(Equal(GPUPort))
					Expect(msg.Src).To(Equal(localPort))
					Expect(msg.SendTime).To(Equal(sim.VTimeInSec(0.0)))
					Expect(msg.TrafficBytes).
						To(Equal(2048))
				})

			GPUPort.EXPECT().
				Send(gomock.Any()).
				Do(func(msg *triosim.TensorMsg) {
					Expect(msg.TensorPkg[0].ID).To(Equal("5"))
					Expect(msg.DstRegionName).To(Equal(localMemRegion.Name))
					Expect(msg.GPUID).To(Equal(0))
					Expect(msg.Purpose).To(Equal("gather"))
					Expect(msg.Dst).To(Equal(localPort))
					Expect(msg.Src).To(Equal(GPUPort))
					Expect(msg.TrafficBytes).
						To(Equal(2048))
				})

			gpuChunks = append(gpuChunks, chunk1)
			gpuChunks = append(gpuChunks, chunk2)
			player.ncclAction = "allgather"
			player.gpuchunks = gpuChunks
			player.playNextReduce()

		})
	})
})
