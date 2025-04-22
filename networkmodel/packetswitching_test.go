package networkmodel

import (
	gomock "github.com/golang/mock/gomock"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	"github.com/syifan/triosim"
	sim "gitlab.com/akita/akita/v3/sim"
)

var _ = Describe("PacketSwitchingNetworkModel", func() {
	var (
		mockCtrl       *gomock.Controller
		model          *PacketSwitchingNetworkModel
		eventScheduler *MockEventScheduler
		src, dst       *MockPort
		timeTeller     *MockTimeTeller
		tensorMsg      *triosim.TensorMsg
	)

	BeforeEach(func() {
		mockCtrl = gomock.NewController(GinkgoT())
		eventScheduler = NewMockEventScheduler(mockCtrl)
		timeTeller = NewMockTimeTeller(mockCtrl)
		model = NewPacketSwitchingNetworkModel(eventScheduler, timeTeller)

		src = NewMockPort(mockCtrl)
		src.EXPECT().Name().Return("src").AnyTimes()
		dst = NewMockPort(mockCtrl)
		dst.EXPECT().Name().Return("dst").AnyTimes()

		model.networkNodes = make(map[string]*NetworkNodes)
		model.networkNodes[src.Name()] = &NetworkNodes{nodetype: "", nodeports: src}
		model.networkNodes[dst.Name()] = &NetworkNodes{nodetype: "", nodeports: dst}

		model.links[src.Name()] = append(model.links[src.Name()], &PSLink{
			link: &Link{
				BytePerSecond: 100,
				Latency:       0.1,
				Left:          src,
				Right:         dst,
			},
			routes: make(map[string]*Route),
		})
		model.links[dst.Name()] = append(model.links[dst.Name()], &PSLink{
			link: &Link{
				BytePerSecond: 100,
				Latency:       0.1,
				Left:          src,
				Right:         dst,
			},
			routes: make(map[string]*Route),
		})

		tensors := make([]triosim.Tensor, 0)
		tensors = append(tensors, triosim.Tensor{})
		tensorMsg = &triosim.TensorMsg{
			MsgMeta: sim.MsgMeta{
				Src:          src,
				Dst:          dst,
				SendTime:     sim.VTimeInSec(0.0),
				TrafficBytes: 100,
				ID:           "1",
			},
			TensorPkg: tensors,
		}

		newRoute := &Route{
			src:          src,
			dst:          dst,
			links:        model.links[dst.Name()],
			msg:          tensorMsg,
			scheduleTime: sim.VTimeInSec(1.1),
		}
		model.routes["1"] = newRoute

	})

	AfterEach(func() {
		mockCtrl.Finish()
	})

	Context("when transferring a single message", func() {

		var (
			testLeft, testRight *MockPort
			testMsg             *triosim.TensorMsg
		)

		BeforeEach(func() {
			testLeft = NewMockPort(mockCtrl)
			testLeft.EXPECT().Name().Return("testLeft").AnyTimes()
			testRight = NewMockPort(mockCtrl)
			testRight.EXPECT().Name().Return("testRight").AnyTimes()

			testMsg = &triosim.TensorMsg{
				MsgMeta: sim.MsgMeta{
					Src:          testLeft,
					Dst:          testRight,
					SendTime:     sim.VTimeInSec(0.0),
					TrafficBytes: 100,
					ID:           "2",
				},
				TensorPkg: make([]triosim.Tensor, 1),
			}
		})

		It("should plugin for ports", func() {
			testLeft.EXPECT().SetConnection(model).Times(1)
			testRight.EXPECT().SetConnection(model).Times(1)

			model.PlugInWithDetails(testLeft, 1, "")
			model.PlugInWithDetails(testRight, 1, "")
		})

		It("should add links for ports", func() {
			model.AddLink(testLeft, testRight, 64/8*1e9, 1e-7)
			PSLink := model.findLinkFromPorts(testLeft, testRight)
			Expect(PSLink).ToNot(BeNil())
		})

		It("should find routes for ports", func() {
			testLeft.EXPECT().SetConnection(model).Times(1)
			testRight.EXPECT().SetConnection(model).Times(1)

			model.PlugInWithDetails(testLeft, 1, "")
			model.PlugInWithDetails(testRight, 1, "")

			model.AddLink(testLeft, testRight, 64/8*1e9, 1e-7)
			PSLink := model.findLinkFromPorts(testLeft, testRight)
			Expect(PSLink).ToNot(BeNil())

			route := model.findRoute(tensorMsg)
			Expect(route).ToNot(BeNil())
			route = model.findRoute(testMsg)
			Expect(route).ToNot(BeNil())
		})

		It("should update progress and bandwidth", func() {
			testLeft.EXPECT().SetConnection(model).Times(1)
			testRight.EXPECT().SetConnection(model).Times(1)

			model.PlugInWithDetails(testLeft, 1, "")
			model.PlugInWithDetails(testRight, 1, "")

			model.AddLink(testLeft, testRight, 64/8*1e9, 1e-7)
			PSLink := model.findLinkFromPorts(testLeft, testRight)
			Expect(PSLink).ToNot(BeNil())

			route := model.findRoute(testMsg)
			Expect(route).ToNot(BeNil())

			timeTeller.EXPECT().CurrentTime().Return(sim.VTimeInSec(0.0)).AnyTimes()
			eventScheduler.EXPECT().Schedule(transferUpdateEvent{
				time:    sim.VTimeInSec(1.25e-08),
				handler: model,
				msg:     testMsg,
			})

			err := model.Send(testMsg)
			Expect(err).To(BeNil())
		})

		It("should schedule an event when a transfer starts", func() {
			timeTeller.EXPECT().CurrentTime().Return(sim.VTimeInSec(0.0)).AnyTimes()
			eventScheduler.EXPECT().Schedule(transferUpdateEvent{
				time:    sim.VTimeInSec(1.1),
				handler: model,
				msg:     tensorMsg,
			})

			err := model.Send(tensorMsg)
			Expect(err).To(BeNil())
		})

		It("should not deliver if the port is busy", func() {
			model.busyNodes[dst.Name()] = true

			err := model.Handle(transferUpdateEvent{
				time:    sim.VTimeInSec(1.1),
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
			err := model.Handle(transferUpdateEvent{
				time:    sim.VTimeInSec(1.1),
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
			err := model.Handle(transferUpdateEvent{
				time:    sim.VTimeInSec(1.1),
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

})
