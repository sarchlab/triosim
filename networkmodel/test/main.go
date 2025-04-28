package main

import (
	"github.com/sarchlab/triosim/networkmodel"
	"github.com/tebeka/atexit"
	"gitlab.com/akita/akita/v3/monitoring"
	"gitlab.com/akita/akita/v3/sim"
)

func main() {
	monitor := monitoring.NewMonitor()

	engine := sim.NewSerialEngine()
	monitor.RegisterEngine(engine)

	freq := sim.Freq(1 * sim.GHz)

	test := NewTest()
	agent1 := NewAgent(engine, freq, "Agent[1]", 1, test)
	agent1.TickLater(0)
	monitor.RegisterComponent(agent1)

	agent2 := NewAgent(engine, freq, "Agent[2]", 1, test)
	agent2.TickLater(0)
	monitor.RegisterComponent(agent2)

	test.RegisterAgent(agent1)
	test.RegisterAgent(agent2)

	test.GenerateMsgs(1000)

	network := networkmodel.NewOpticalNetworkModel(engine, engine, 1, 20)
	network.PlugIn(agent1.AgentPorts[0], 1)
	network.PlugIn(agent2.AgentPorts[0], 1)

	network.AddWaveGuide([]sim.Port{
		agent1.AgentPorts[0],
		agent2.AgentPorts[0],
	}, 1, 1<<30)

	monitor.StartServer()

	err := engine.Run()
	if err != nil {
		panic(err)
	}

	test.MustHaveReceivedAllMsgs()
	test.ReportBandwidthAchieved(engine.CurrentTime())
	atexit.Exit(0)
}
