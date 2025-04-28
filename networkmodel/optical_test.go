package networkmodel

import (
	gomock "github.com/golang/mock/gomock"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	"github.com/sarchlab/triosim"
	sim "gitlab.com/akita/akita/v3/sim"
)

var _ = Describe("OpticalNetworkModel", func() {
	var (
		mockCtrl             *gomock.Controller
		model                *OpticalNetworkModel
		eventScheduler       *MockEventScheduler
		src, dst, src1, dst1 *MockPort
		timeTeller           *MockTimeTeller
		latency              float64
		bytePerSecond        float64
	)

	BeforeEach(func() {
		mockCtrl = gomock.NewController(GinkgoT())
		eventScheduler = NewMockEventScheduler(mockCtrl)
		timeTeller = NewMockTimeTeller(mockCtrl)
		model = NewOpticalNetworkModel(eventScheduler, timeTeller, 1, 20)

		src = NewMockPort(mockCtrl)
		src.EXPECT().Name().Return("GPU0Port").AnyTimes()
		dst = NewMockPort(mockCtrl)
		dst.EXPECT().Name().Return("GPU1Port").AnyTimes()

		src1 = NewMockPort(mockCtrl)
		src1.EXPECT().Name().Return("src1").AnyTimes()
		dst1 = NewMockPort(mockCtrl)
		dst1.EXPECT().Name().Return("dst1").AnyTimes()

	})

	AfterEach(func() {
		mockCtrl.Finish()
	})

	Context("when transferring a single message and find existing waveguide", func() {
		var (
			tensorMsg *triosim.TensorMsg
		)

		BeforeEach(func() {
			tensorMsg = &triosim.TensorMsg{
				MsgMeta: sim.MsgMeta{
					Src:          src,
					Dst:          dst,
					SendTime:     sim.VTimeInSec(0.0),
					TrafficBytes: 100,
				},
				TensorPkg: make([]triosim.Tensor, 1),
			}
			// model.AddWaveGuide([]sim.Port{src, dst}, 1000, 1.0)
			model.InitHardwareNetwork("mesh", 1, 2)
			model.InitLogicalNetwork("ring", 1, 2)
			latency = 20 * 1e-9
			bytePerSecond = 64 * 1e9
		})

		It("should schedule an event when a transfer starts", func() {
			timeTeller.EXPECT().CurrentTime().Return(sim.VTimeInSec(0.0))
			transferTime := sim.VTimeInSec(float64(tensorMsg.Meta().TrafficBytes) / bytePerSecond)
			time := transferTime + sim.VTimeInSec(latency)
			eventScheduler.EXPECT().Schedule(transferUpdateEvent{
				time:    time,
				handler: model,
				msg:     tensorMsg,
			})

			err := model.Send(tensorMsg)

			Expect(err).To(BeNil())
		})

		It("should not deliver if the port is busy", func() {
			model.busyNodes[dst.Name()] = true
			transferTime := sim.VTimeInSec(float64(tensorMsg.Meta().TrafficBytes) / bytePerSecond)
			time := transferTime + sim.VTimeInSec(latency)
			err := model.Handle(transferUpdateEvent{
				time:    time,
				handler: model,
				msg:     tensorMsg,
			})

			Expect(err).To(BeNil())
		})

		It("should deliver the message when the transfer is completed", func() {
			timeTeller.EXPECT().
				CurrentTime().
				Return(sim.VTimeInSec(1.1)).
				AnyTimes()

			dst.EXPECT().Recv(tensorMsg).Return(nil)
			transferTime := sim.VTimeInSec(float64(tensorMsg.Meta().TrafficBytes) / bytePerSecond)
			time := transferTime + sim.VTimeInSec(latency)
			err := model.Handle(transferUpdateEvent{
				time:    time,
				handler: model,
				msg:     tensorMsg,
			})

			Expect(err).To(BeNil())
			Expect(tensorMsg.Meta().RecvTime).To(Equal(sim.VTimeInSec(1.1)))
		})

		It("should mark the destination as busy if the dst is busy", func() {
			timeTeller.EXPECT().
				CurrentTime().
				Return(sim.VTimeInSec(1.1)).
				AnyTimes()

			dst.EXPECT().Recv(tensorMsg).Return(&sim.SendError{})
			transferTime := sim.VTimeInSec(float64(tensorMsg.Meta().TrafficBytes) / bytePerSecond)
			time := transferTime + sim.VTimeInSec(latency)
			err := model.Handle(transferUpdateEvent{
				time:    time,
				handler: model,
				msg:     tensorMsg,
			})

			Expect(err).To(BeNil())
			Expect(model.busyNodes).To(HaveKey(dst.Name()))
		})

		It("should deliver pending messages when the port is free", func() {
			model.pendingDelivery[dst.Name()] = []sim.Msg{tensorMsg}
			model.busyNodes[dst.Name()] = true

			dst.EXPECT().Recv(tensorMsg).Return(nil)

			model.NotifyAvailable(0, dst)

			Expect(model.busyNodes).ToNot(HaveKey(dst.Name()))
			Expect(model.pendingDelivery[dst.Name()]).To(BeEmpty())
		})

		It("should still not deliver if the port is busy", func() {
			model.pendingDelivery[dst.Name()] = []sim.Msg{tensorMsg}
			model.busyNodes[dst.Name()] = true

			dst.EXPECT().Recv(tensorMsg).Return(&sim.SendError{})

			model.NotifyAvailable(0, dst)

			Expect(model.busyNodes).To(HaveKey(dst.Name()))
			Expect(model.pendingDelivery[dst.Name()]).To(HaveLen(1))
		})
	})

	Context("when transferring a single message and no existing waveguide", func() {
		var (
			tensorMsg *triosim.TensorMsg
			// tensorMsg0 *triosim.TensorMsg
			// tensorMsg1 *triosim.TensorMsg
			// tensorMsg2 *triosim.TensorMsg
		)

		BeforeEach(func() {
			tensorMsg = &triosim.TensorMsg{
				MsgMeta: sim.MsgMeta{
					Src:          src,
					Dst:          dst,
					SendTime:     sim.VTimeInSec(0.0),
					TrafficBytes: 100,
				},
				TensorPkg: make([]triosim.Tensor, 1),
			}
			// tensorMsg0 = &triosim.TensorMsg{
			// 	MsgMeta: sim.MsgMeta{
			// 		Src:          dst,
			// 		Dst:          src,
			// 		SendTime:     sim.VTimeInSec(10.0),
			// 		TrafficBytes: 150,
			// 	},
			// 	TensorPkg: make([]triosim.Tensor, 1),
			// }
			// tensorMsg1 = &triosim.TensorMsg{
			// 	MsgMeta: sim.MsgMeta{
			// 		Src:          src,
			// 		Dst:          dst1,
			// 		SendTime:     sim.VTimeInSec(10.0),
			// 		TrafficBytes: 150,
			// 	},
			// 	TensorPkg: make([]triosim.Tensor, 1),
			// }
			// tensorMsg2 = &triosim.TensorMsg{
			// 	MsgMeta: sim.MsgMeta{
			// 		Src:          src1,
			// 		Dst:          dst1,
			// 		SendTime:     sim.VTimeInSec(10.0),
			// 		TrafficBytes: 150,
			// 	},
			// 	TensorPkg: make([]triosim.Tensor, 1),
			// }
			// model.AddWaveGuide([]sim.Port{src, dst}, 1000, 1.0)
			model.InitHardwareNetwork("mesh", 1, 2)
			model.InitLogicalNetwork("ring", 1, 2)
			latency = 20 * 1e-9
			bytePerSecond = 64 * 1e9
		})

		It("should schedule a establish event when a transfer starts", func() {
			//establishWaveGuideEvent we initliaze the establish processe before the transfer starts
			//by InitHardwareNetwork and InitLogicalNetwork
			timeTeller.EXPECT().CurrentTime().Return(sim.VTimeInSec(0.0))
			transferTime := sim.VTimeInSec(float64(tensorMsg.Meta().TrafficBytes) / bytePerSecond)
			time := transferTime + sim.VTimeInSec(latency)
			// eventScheduler.EXPECT().Schedule(establishWaveGuideEvent{
			eventScheduler.EXPECT().Schedule(transferUpdateEvent{
				time:    time,
				handler: model,
				msg:     tensorMsg,
			})

			err := model.Send(tensorMsg)

			Expect(err).To(BeNil())
			// Expect(len(model.inflightEstablishTransactions[tensorMsg.Meta().Src])).To(Equal(1))
			// Expect(len(model.inflightEstablishTransactions[tensorMsg.Meta().Dst])).To(Equal(1))
			Expect(len(model.inflightEstablishTransactions[tensorMsg.Meta().Src])).To(Equal(0))
			Expect(len(model.inflightEstablishTransactions[tensorMsg.Meta().Dst])).To(Equal(0))
		})
		//in the current implementation, we initliaze the establish processe before the transfer starts
		//by InitHardwareNetwork and InitLogicalNetwork, so we skip this test
		//nolint:lll
		// It("should establish waveguide and schedule transferUpdate event", func() {
		// 	timeTeller.EXPECT().CurrentTime().Return(sim.VTimeInSec(20.0))
		// 	eventScheduler.EXPECT().Schedule(transferUpdateEvent{
		// 		time:    sim.VTimeInSec(21.1),
		// 		handler: model,
		// 		msg:     tensorMsg,
		// 	})
		// 	event := establishWaveGuideEvent{
		// 		time:    sim.VTimeInSec(20.0),
		// 		handler: model,
		// 		msg:     tensorMsg,
		// 	}
		// 	t := inflightEstablishTransaction{e: event,
		// 		ports: []sim.Port{tensorMsg.Meta().Src, tensorMsg.Meta().Dst}}
		// 	model.inflightEstablishTransactions[tensorMsg.Meta().Src] = append(model.inflightEstablishTransactions[tensorMsg.Meta().Src], &t)
		// 	model.inflightEstablishTransactions[tensorMsg.Meta().Dst] = append(model.inflightEstablishTransactions[tensorMsg.Meta().Dst], &t)

		// 	err := model.Handle(event)
		// 	Expect(err).To(BeNil())
		// 	Expect(len(model.waveGuides[tensorMsg.Meta().Src])).To(Equal(1))
		// 	Expect(len(model.waveGuides[tensorMsg.Meta().Dst])).To(Equal(1))
		// 	Expect(model.wgCounts).To(Equal(1))
		// 	Expect(len(model.inflightEstablishTransactions[tensorMsg.Meta().Src])).To(Equal(0))
		// 	Expect(len(model.inflightEstablishTransactions[tensorMsg.Meta().Dst])).To(Equal(0))
		// })
		// //nolint:lll
		// It("should not schedule another establishWaveGuideEvent if the requested wg is being established", func() {
		// 	t := inflightEstablishTransaction{e: establishWaveGuideEvent{},
		// 		ports: []sim.Port{tensorMsg.Meta().Src, tensorMsg.Meta().Dst}}

		// 	model.inflightEstablishTransactions[tensorMsg.Meta().Src] =
		// 	append(model.inflightEstablishTransactions[tensorMsg.Meta().Src], &t)
		// 	model.inflightEstablishTransactions[tensorMsg.Meta().Dst] =
		// 	append(model.inflightEstablishTransactions[tensorMsg.Meta().Dst], &t)
		// 	timeTeller.EXPECT().CurrentTime().Return(sim.VTimeInSec(0.0))

		// 	err := model.Send(tensorMsg0)

		// 	Expect(err).To(BeNil())
		// 	Expect(model.inflightEstablishTransactions[tensorMsg.Meta().Src][0].msgs).To(Equal([]sim.Msg{tensorMsg0}))
		// 	Expect(model.inflightEstablishTransactions[tensorMsg.Meta().Dst][0].msgs).To(Equal([]sim.Msg{tensorMsg0}))
		// })
		// //nolint:lll
		// It("should establish waveguide and schedule two transferUpdate events", func() {
		// 	timeTeller.EXPECT().CurrentTime().Return(sim.VTimeInSec(20.0))
		// 	eventScheduler.EXPECT().Schedule(transferUpdateEvent{
		// 		time:    sim.VTimeInSec(21.1),
		// 		handler: model,
		// 		msg:     tensorMsg,
		// 	})
		// 	eventScheduler.EXPECT().Schedule(transferUpdateEvent{
		// 		time:    sim.VTimeInSec(21.15),
		// 		handler: model,
		// 		msg:     tensorMsg0,
		// 	})
		// 	event := establishWaveGuideEvent{
		// 		time:    sim.VTimeInSec(20.0),
		// 		handler: model,
		// 		msg:     tensorMsg,
		// 	}
		// 	t := inflightEstablishTransaction{e: event,
		// 		ports: []sim.Port{tensorMsg.Meta().Src, tensorMsg.Meta().Dst}}
		// 	t.msgs = append(t.msgs, tensorMsg0)
		// 	model.inflightEstablishTransactions[tensorMsg.Meta().Src] = append(model.inflightEstablishTransactions[tensorMsg.Meta().Src], &t)
		// 	model.inflightEstablishTransactions[tensorMsg.Meta().Dst] = append(model.inflightEstablishTransactions[tensorMsg.Meta().Dst], &t)

		// 	err := model.Handle(event)
		// 	Expect(err).To(BeNil())
		// 	Expect(len(model.waveGuides[tensorMsg.Meta().Src])).To(Equal(1))
		// 	Expect(len(model.waveGuides[tensorMsg.Meta().Dst])).To(Equal(1))
		// 	Expect(model.wgCounts).To(Equal(1))
		// 	Expect(len(model.inflightEstablishTransactions[tensorMsg.Meta().Src])).To(Equal(0))
		// 	Expect(len(model.inflightEstablishTransactions[tensorMsg.Meta().Dst])).To(Equal(0))
		// })
		// //nolint:lll
		// It("should not schedule another establishWaveGuideEvent if no available wgs", func() {
		// 	t := inflightEstablishTransaction{e: establishWaveGuideEvent{},
		// 		ports: []sim.Port{tensorMsg.Meta().Src, tensorMsg.Meta().Dst}}
		// 	model.inflightEstablishTransactions[tensorMsg.Meta().Src] = append(model.inflightEstablishTransactions[tensorMsg.Meta().Src], &t)
		// 	model.inflightEstablishTransactions[tensorMsg.Meta().Dst] = append(model.inflightEstablishTransactions[tensorMsg.Meta().Dst], &t)
		// 	timeTeller.EXPECT().CurrentTime().Return(sim.VTimeInSec(10.0))

		// 	err1 := model.Send(tensorMsg1)

		// 	Expect(err1).To(Equal(sim.NewSendError()))
		// })

		// It("should not schedule another establishWaveGuideEvent if no available wgs", func() {
		// 	wg := OpticalWaveGuide{
		// 		Ports:         []sim.Port{tensorMsg.Meta().Src, tensorMsg.Meta().Dst},
		// 		BytePerSecond: 1000,
		// 		Latency:       1.0}
		// 	model.waveGuides[tensorMsg.Meta().Src] = append(model.waveGuides[tensorMsg.Meta().Src], &wg)
		// 	model.waveGuides[tensorMsg.Meta().Dst] = append(model.waveGuides[tensorMsg.Meta().Dst], &wg)
		// 	model.pendingDelivery[tensorMsg.Meta().Dst.Name()] = append(
		// 		model.pendingDelivery[tensorMsg.Meta().Dst.Name()],
		// 		tensorMsg)
		// 	timeTeller.EXPECT().CurrentTime().Return(sim.VTimeInSec(0.0))

		// 	err1 := model.Send(tensorMsg1)

		// 	Expect(err1).To(Equal(sim.NewSendError()))
		// 	Expect(model.busy).To(Equal(true))
		// })

		// It("should destroy a wg and schedule another establishWaveGuideEvent", func() {
		// 	wg := OpticalWaveGuide{
		// 		Ports:         []sim.Port{tensorMsg.Meta().Src, tensorMsg.Meta().Dst},
		// 		BytePerSecond: 1000,
		// 		Latency:       1.0}
		// 	model.waveGuides[tensorMsg.Meta().Src] = append(model.waveGuides[tensorMsg.Meta().Src], &wg)
		// 	model.waveGuides[tensorMsg.Meta().Dst] = append(model.waveGuides[tensorMsg.Meta().Dst], &wg)
		// 	model.wgCounts = 1
		// 	timeTeller.EXPECT().CurrentTime().Return(sim.VTimeInSec(0.0))
		// 	eventScheduler.EXPECT().Schedule(establishWaveGuideEvent{
		// 		time:    sim.VTimeInSec(20.0),
		// 		handler: model,
		// 		msg:     tensorMsg1,
		// 	})

		// 	err1 := model.Send(tensorMsg1)

		// 	Expect(err1).To(BeNil())
		// 	Expect(len(model.inflightEstablishTransactions[tensorMsg1.Meta().Src])).To(Equal(1))
		// 	Expect(len(model.inflightEstablishTransactions[tensorMsg1.Meta().Dst])).To(Equal(1))
		// 	Expect(len(model.waveGuides[tensorMsg.Meta().Src])).To(Equal(0))
		// 	Expect(len(model.waveGuides[tensorMsg.Meta().Dst])).To(Equal(0))
		// 	Expect(model.wgCounts).To(Equal(0))

		// })

		// It("should destroy a wg and schedule another establishWaveGuideEvent", func() {
		// 	wg := OpticalWaveGuide{
		// 		Ports:         []sim.Port{tensorMsg1.Meta().Src, tensorMsg1.Meta().Dst},
		// 		BytePerSecond: 1000,
		// 		Latency:       1.0}
		// 	model.waveGuides[tensorMsg1.Meta().Src] = append(model.waveGuides[tensorMsg1.Meta().Src], &wg)
		// 	model.waveGuides[tensorMsg1.Meta().Dst] = append(model.waveGuides[tensorMsg1.Meta().Dst], &wg)
		// 	model.wgCounts = 1
		// 	timeTeller.EXPECT().CurrentTime().Return(sim.VTimeInSec(0.0))
		// 	eventScheduler.EXPECT().Schedule(establishWaveGuideEvent{
		// 		time:    sim.VTimeInSec(20.0),
		// 		handler: model,
		// 		msg:     tensorMsg2,
		// 	})

		// 	err1 := model.Send(tensorMsg2)

		// 	Expect(err1).To(BeNil())
		// 	Expect(len(model.inflightEstablishTransactions[tensorMsg2.Meta().Src])).To(Equal(1))
		// 	Expect(len(model.inflightEstablishTransactions[tensorMsg2.Meta().Dst])).To(Equal(1))
		// 	Expect(len(model.waveGuides[tensorMsg1.Meta().Src])).To(Equal(0))
		// 	Expect(len(model.waveGuides[tensorMsg1.Meta().Dst])).To(Equal(0))
		// 	Expect(model.wgCounts).To(Equal(0))
		// })
	})

})
