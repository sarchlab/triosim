package networkmodel

import (
	"fmt"
	"strconv"

	"gitlab.com/akita/akita/v3/sim"
)

// A OpticalWaveGuide is a wave guide in the network. It is similar to a link,
// but can connect with more than two ports.
type OpticalWaveGuide struct {
	BytePerSecond float64
	Latency       sim.VTimeInSec
	Ports         []sim.Port
}

type establishWaveGuideEvent struct {
	time    sim.VTimeInSec
	handler sim.Handler
	msg     sim.Msg
}

func (e establishWaveGuideEvent) Time() sim.VTimeInSec {
	return e.time
}

func (e establishWaveGuideEvent) Handler() sim.Handler {
	return e.handler
}

func (e establishWaveGuideEvent) IsSecondary() bool {
	return false
}

type inflightEstablishTransaction struct {
	e    establishWaveGuideEvent
	msgs []sim.Msg
	// ports []sim.Port
}

// A OpticalNetworkModel is a simple network model that can estimate
// the arrival time of transfers.
type OpticalNetworkModel struct {
	sim.HookableBase
	sim.EventScheduler
	sim.TimeTeller

	nodes      map[string]sim.Port
	waveGuides map[sim.Port][]*OpticalWaveGuide

	busyNodes                     map[string]bool
	pendingDelivery               map[string][]sim.Msg
	maxNumWaveGuidesPerNode       int
	busy                          bool
	establishLatency              sim.VTimeInSec
	inflightEstablishTransactions map[sim.Port][]*inflightEstablishTransaction
	wgCounts                      int
	hardwareLinks                 map[string]map[string]*Link
	//it's waveguide actually, use hardwareLink to represent it now, will change it later
	logicalLinks      map[string]map[string]*Link
	logicalLinksCount map[string]int
	maxNumPorts       int
	logicalChannels   map[string]map[string]*Link
	totalHops         int
}

// NewOpticalNetworkModel creates a new OpticalNetworkModel.
func NewOpticalNetworkModel(
	es sim.EventScheduler,
	tt sim.TimeTeller,
	maxNumWaveGuidesPerNode int,
	establishLatency sim.VTimeInSec,
) *OpticalNetworkModel {
	m := &OpticalNetworkModel{
		EventScheduler:                es,
		TimeTeller:                    tt,
		nodes:                         make(map[string]sim.Port),
		busyNodes:                     make(map[string]bool),
		pendingDelivery:               make(map[string][]sim.Msg),
		waveGuides:                    make(map[sim.Port][]*OpticalWaveGuide),
		inflightEstablishTransactions: make(map[sim.Port][]*inflightEstablishTransaction),
		maxNumWaveGuidesPerNode:       maxNumWaveGuidesPerNode,
		establishLatency:              establishLatency,
		logicalLinksCount:             make(map[string]int),
		maxNumPorts:                   maxNumWaveGuidesPerNode, //lytest fix it later
		hardwareLinks:                 make(map[string]map[string]*Link),
		logicalLinks:                  make(map[string]map[string]*Link),
		logicalChannels:               make(map[string]map[string]*Link),
	}

	return m
}

// PlugIn plugs a port into the network.
func (m *OpticalNetworkModel) PlugIn(port sim.Port, bufSize int) {
	m.nodes[port.Name()] = port
	port.SetConnection(m)
}

// Unplug removes a port from the network.
func (m *OpticalNetworkModel) Unplug(port sim.Port) {
	delete(m.nodes, port.Name())
}

// NotifyAvailable notifies the network that the port is available to send
// messages.
func (m *OpticalNetworkModel) NotifyAvailable(
	now sim.VTimeInSec,
	port sim.Port,
) {
	pendingDelivery := m.pendingDelivery[port.Name()]
	var src sim.Port

	for len(pendingDelivery) > 0 {
		msg := pendingDelivery[0]
		src = msg.Meta().Src
		err := port.Recv(msg)
		if err != nil {
			break
		}

		pendingDelivery = pendingDelivery[1:]
	}

	m.pendingDelivery[port.Name()] = pendingDelivery

	if len(pendingDelivery) == 0 {
		delete(m.busyNodes, port.Name())
		if _, ok := m.busyNodes[src.Name()]; !ok {
			if m.busy {
				for _, port := range m.nodes {
					port.NotifyAvailable(now)
				}
			}
		}
	}
}

func (m *OpticalNetworkModel) InitHardwareNetwork(networkType string, numRows int, numCols int) (int, int) {
	//may need to initialize the hardwareLinks first
	if networkType == "mesh" {
		for r := 0; r < numRows; r++ {
			for c := 0; c < numCols; c++ {
				currentIndex := r*numCols + c
				currentPortName := "GPU" + strconv.Itoa(currentIndex) + "Port"
				currentPort := m.nodes[currentPortName]
				// Connect to the right neighbor
				if c < numCols-1 {
					rightIndex := currentIndex + 1
					rightPortName := "GPU" + strconv.Itoa(rightIndex) + "Port"
					rightPort := m.nodes[rightPortName]
					hardwareLink := m.initLink(currentPort, rightPort)
					if m.hardwareLinks[currentPortName] == nil {
						m.hardwareLinks[currentPortName] = make(map[string]*Link)
					}

					m.hardwareLinks[currentPortName][rightPortName] = hardwareLink
					fmt.Println(currentPortName, rightPortName)
					//bidirectional, we think it's a link
					if m.hardwareLinks[rightPortName] == nil {
						m.hardwareLinks[rightPortName] = make(map[string]*Link)
					}

					m.hardwareLinks[rightPortName][currentPortName] = hardwareLink
				}

				// Connect to the bottom neighbor
				if r < numRows-1 {
					downIndex := currentIndex + numCols
					downPortName := "GPU" + strconv.Itoa(downIndex) + "Port"
					downPort := m.nodes[downPortName]
					hardwareLink := m.initLink(currentPort, downPort)
					if m.hardwareLinks[currentPortName] == nil {
						m.hardwareLinks[currentPortName] = make(map[string]*Link)
					}

					m.hardwareLinks[currentPortName][downPortName] = hardwareLink
					fmt.Println(currentPortName, downPortName)
					//bidirectional, we think it's a link
					if m.hardwareLinks[downPortName] == nil {
						m.hardwareLinks[downPortName] = make(map[string]*Link)
					}

					m.hardwareLinks[downPortName][currentPortName] = hardwareLink
				}
			}
		}
	} else {
		panic("Invalid hardware network type, only 2D mesh is supported now")
	}
	return numRows, numCols
}

func (m *OpticalNetworkModel) InitLogicalNetwork(networkType string, numRows int, numCols int) {
	switch networkType {
	case "ring": //lytest debug to make sure the ring is correct
		m.buildRing(numRows, numCols)
		m.buildRemoteNetwork(numRows, numCols)
	case "butterfly":
		m.buildButterfly(numRows, numCols)
		m.buildRemoteNetwork(numRows, numCols)
	default:
		panic("Invalid logical network type, only ring is supported now")
	}
}

func (m *OpticalNetworkModel) buildRemoteNetwork(numRows int, numCols int) {
	for i := 0; i < numRows; i++ {
		for j := 0; j < numCols; j++ {
			currentIndex := i*numCols + j
			currentPortName := "GPU" + strconv.Itoa(currentIndex) + "Port"
			m.buildNetworkChannels(currentPortName, "RemotePort")
		}
	}
}

func (m *OpticalNetworkModel) buildRing(numRows int, numCols int) {
	ringOrder := m.getTraversalOrder(numRows, numCols)
	// Close the ring by appending the first node again
	ringOrder = append(ringOrder, ringOrder[0])
	fmt.Println(ringOrder)
	for i := 0; i < len(ringOrder)-1; i++ {
		fromPortName := "GPU" + strconv.Itoa(ringOrder[i]) + "Port"
		toPortName := "GPU" + strconv.Itoa(ringOrder[i+1]) + "Port"
		m.buildNetworkChannels(fromPortName, toPortName)
	}

	ringOrder = ringOrder[:len(ringOrder)-1]
	//by now, each chip only one port (input and output)
	if numRows == 1 {
		lastElement := ringOrder[len(ringOrder)-2]
		for lastElement%numCols != 0 {
			ringOrder = append(ringOrder, lastElement)
			lastElement -= 1
		}
	}

	if numRows%2 == 1 && numRows != 1 {
		lastElement := ringOrder[len(ringOrder)-1]
		upperNeighbor := lastElement - numCols
		// Keep moving left until we reach the first column
		for upperNeighbor%numCols != 0 {
			ringOrder = append(ringOrder, upperNeighbor)
			upperNeighbor -= 1
		}
	}
	// Close the ring by appending the first node again
	ringOrder = append(ringOrder, ringOrder[0])
	fmt.Println(ringOrder)
	//by now, each chip in the first row except the last one (in the first row) has two ports
	for i := 0; i < len(ringOrder)-1; i++ {
		fromPortName := "GPU" + strconv.Itoa(ringOrder[i]) + "Port"
		toPortName := "GPU" + strconv.Itoa(ringOrder[i+1]) + "Port"
		m.buildNetworkLinks(fromPortName, toPortName)
	}
}

func (m *OpticalNetworkModel) getTraversalOrder(numRows int, numCols int) []int {
	total := numRows * numCols
	// visited keeps track of grid cells we've visited.
	visited := make([][]bool, numRows)
	for i := 0; i < numRows; i++ {
		visited[i] = make([]bool, numCols)
	}

	var path []int
	// directions in order: right, down, left, up.
	directions := []struct{ dr, dc int }{
		{0, 1},  // right
		{1, 0},  // down
		{0, -1}, // left
		{-1, 0}, // up
	}

	var dfs func(r, c int) bool
	dfs = func(r, c int) bool {
		// Mark this cell as visited and record its index.
		visited[r][c] = true
		path = append(path, r*numCols+c)

		// If we've visited all cells, we're done.
		if len(path) == total {
			return true
		}

		// Try each neighbor in the fixed order.
		for _, d := range directions {
			nr, nc := r+d.dr, c+d.dc
			if nr >= 0 && nr < numRows && nc >= 0 && nc < numCols && !visited[nr][nc] {
				if dfs(nr, nc) {
					return true
				}
			}
		}

		// Backtrack: unmark this cell and remove it from the path.
		visited[r][c] = false
		path = path[:len(path)-1]
		return false
	}

	dfs(0, 0)
	return path
}

func (m *OpticalNetworkModel) buildButterfly(numRows int, numCols int) {
	oneLineOrder := m.getTraversalOrder(numRows, numCols)
	fmt.Println("butterfly", oneLineOrder)
	// Iterating over log₂(N) stages
	N := len(oneLineOrder)
	for stage := 0; (1 << stage) < N; stage++ {
		bitMask := 1 << stage // Compute the bit mask

		for i := 0; i < N; i++ {
			neighbor := i ^ bitMask           // XOR to find connected node
			if neighbor < N && i < neighbor { // Ensure we stay within bounds and process each connection once
				// Connect nodes bidirectionally
				srcPortName := "GPU" + strconv.Itoa(i) + "Port"
				dstPortName := "GPU" + strconv.Itoa(neighbor) + "Port"
				fmt.Println("butterfly", srcPortName, dstPortName)
				m.buildNetworkChannels(srcPortName, dstPortName)
				fmt.Println("butterfly--buildNetworkChannels")
				passbyLinks := m.getPassbyLinksOrChannels(srcPortName, dstPortName, "link")
				fmt.Println("butterfly--getPassbyLinksOrChannels")
				for j := 0; j < len(passbyLinks)-1; j++ {
					m.buildNetworkLinks(passbyLinks[j], passbyLinks[j+1])
				}
			}
		}
	}
}

func (m *OpticalNetworkModel) getPassbyLinksOrChannels(srcPortName, dstPortName, caseType string) []string {
	// Queue for BFS.
	queue := []string{srcPortName}
	// visited tracks visited nodes.
	visited := make(map[string]bool)
	visited[srcPortName] = true
	// parent maps a node to its predecessor (to reconstruct the path).
	parent := make(map[string]string)
	for len(queue) > 0 {
		// Dequeue the front.
		current := queue[0]
		queue = queue[1:]
		// If destination is reached, reconstruct the path.
		if current == dstPortName {
			return reconstructPath(parent, srcPortName, dstPortName)
		}

		var neighbors []string
		switch caseType {
		case "link":
			neighbors = m.getValidLinkNeighbors(current, visited)
		case "channel":
			neighbors = m.getValidChannelNeighbors(current, visited)
		default:
			panic("Unknown caseType: must be 'link' or 'channel'")
		}

		for _, neighbor := range neighbors {
			visited[neighbor] = true
			parent[neighbor] = current
			queue = append(queue, neighbor)
		}
	}
	// No path found.
	panic("No path found, check the topology building process")
	// return nil
}

func (m *OpticalNetworkModel) getValidLinkNeighbors(current string, visited map[string]bool) []string {
	var neighbors []string
	for neighbor := range m.hardwareLinks[current] {
		if visited[neighbor] {
			continue
		}
		if m.logicalLinksCount[current] < m.maxNumPorts && m.logicalLinksCount[neighbor] < m.maxNumPorts {
			neighbors = append(neighbors, neighbor)
		}
		// else: fmt.Println("logical link count exceeds the maxNumPorts, skip this link")
	}
	return neighbors
}

func (m *OpticalNetworkModel) getValidChannelNeighbors(current string, visited map[string]bool) []string {
	var neighbors []string
	for neighbor := range m.logicalChannels[current] {
		if !visited[neighbor] && neighbor != "RemotePort" {
			neighbors = append(neighbors, neighbor)
		}
	}
	return neighbors
}

func reconstructPath(parent map[string]string, src, dst string) []string {
	var path []string
	// Backtrack from dst to src.
	for current := dst; current != src; current = parent[current] {
		path = append(path, current)
	}
	path = append(path, src)
	reverse(path)
	return path
}

// reverse reverses a slice of strings in place.
func reverse(path []string) {
	for i, j := 0, len(path)-1; i < j; i, j = i+1, j-1 {
		path[i], path[j] = path[j], path[i]
	}
}

func (m *OpticalNetworkModel) buildNetworkLinks(fromPortName string, toPortName string) {
	logicalLink := m.initLink(m.nodes[fromPortName], m.nodes[toPortName])
	//double check hardware link exists
	fmt.Println(fromPortName, toPortName)
	if _, ok := m.hardwareLinks[fromPortName][toPortName]; !ok {
		panic("Hardware link does not exist, check the topology building process")
	}
	//double check chip ports number exceed the maxNumPorts
	if m.logicalLinksCount[fromPortName] > m.maxNumPorts || m.logicalLinksCount[toPortName] > m.maxNumPorts {
		panic("chip ports number exceeds the maxNumPorts, check the topology building process")
	}
	if m.logicalLinks[fromPortName] == nil {
		m.logicalLinks[fromPortName] = make(map[string]*Link)
	}
	m.logicalLinks[fromPortName][toPortName] = logicalLink
	if m.logicalLinks[toPortName] == nil {
		m.logicalLinks[toPortName] = make(map[string]*Link)
	}
	m.logicalLinks[toPortName][fromPortName] = logicalLink //bidirectional, we think it's a link
	m.logicalLinksCount[fromPortName]++
	m.logicalLinksCount[toPortName]++
	fmt.Println("buildNetworkLinks", fromPortName, toPortName)
	fmt.Println(m.logicalLinksCount[fromPortName], m.logicalLinksCount[toPortName])
}

func (m *OpticalNetworkModel) buildNetworkChannels(fromPortName string, toPortName string) {
	logicalChannel := m.initLink(m.nodes[fromPortName], m.nodes[toPortName])
	if m.logicalChannels[fromPortName] == nil {
		m.logicalChannels[fromPortName] = make(map[string]*Link)
	}
	m.logicalChannels[fromPortName][toPortName] = logicalChannel

	if m.logicalChannels[toPortName] == nil {
		m.logicalChannels[toPortName] = make(map[string]*Link)
	}
	m.logicalChannels[toPortName][fromPortName] = logicalChannel //bidirectional, we think it's a channel
}

// AddWaveGuide adds a wave guide to the network to connect with the given
// ports.
func (m *OpticalNetworkModel) AddWaveGuide(
	ports []sim.Port,
	bytePerSecond float64,
	latency sim.VTimeInSec,
) {
	wg := OpticalWaveGuide{
		Ports:         ports,
		BytePerSecond: bytePerSecond,
		Latency:       latency,
	}
	for _, port := range ports {
		m.waveGuides[port] = append(m.waveGuides[port], &wg)
	}
}

// WaveGuideCounts returns how many waveguides are in use
func (m *OpticalNetworkModel) WaveGuideCounts() int {
	return m.wgCounts
}

// Handle checks if the transfers are completed.
func (m *OpticalNetworkModel) Handle(e sim.Event) error {
	switch e := e.(type) {
	case transferUpdateEvent:
		return m.handleTransferUpdateEvent(e)
	case establishWaveGuideEvent:
		return m.handleEstablishWaveGuideEvent(e)
	default:
		panic("unknown event type")
	}
}

func (m *OpticalNetworkModel) handleTransferUpdateEvent(
	e transferUpdateEvent,
) error {
	msg := e.msg

	if _, busy := m.busyNodes[msg.Meta().Dst.Name()]; busy {
		m.pendingDelivery[msg.Meta().Dst.Name()] = append(
			m.pendingDelivery[msg.Meta().Dst.Name()],
			msg,
		)
		return nil
	}

	msg.Meta().RecvTime = m.CurrentTime()
	err := msg.Meta().Dst.Recv(msg)
	if err != nil {
		m.busyNodes[msg.Meta().Dst.Name()] = true
		m.pendingDelivery[msg.Meta().Dst.Name()] = append(
			m.pendingDelivery[msg.Meta().Dst.Name()],
			msg,
		)
	}

	return nil
}

func (m *OpticalNetworkModel) handleEstablishWaveGuideEvent(
	e establishWaveGuideEvent,
) error {
	now := m.CurrentTime()
	msg := e.msg
	wg := OpticalWaveGuide{
		Ports:         []sim.Port{msg.Meta().Src, msg.Meta().Dst},
		BytePerSecond: 64 * 1e9,
		Latency:       20 * 1e-9}
	m.waveGuides[msg.Meta().Src] = append(m.waveGuides[msg.Meta().Src], &wg)
	m.waveGuides[msg.Meta().Dst] = append(m.waveGuides[msg.Meta().Dst], &wg)
	m.wgCounts++

	transferTime := sim.VTimeInSec(float64(msg.Meta().TrafficBytes) /
		wg.BytePerSecond)
	m.Schedule(transferUpdateEvent{
		time:    now + wg.Latency + transferTime,
		handler: m,
		msg:     msg,
	})

	otherMsgs := m.removeInflightEstablishTransaction(e)
	for _, omsg := range otherMsgs {
		transferTime := sim.VTimeInSec(float64(omsg.Meta().TrafficBytes) /
			wg.BytePerSecond)
		m.Schedule(transferUpdateEvent{
			time:    now + wg.Latency + transferTime,
			handler: m,
			msg:     omsg,
		})
	}

	return nil
}

func (m *OpticalNetworkModel) removeInflightEstablishTransaction(e establishWaveGuideEvent) []sim.Msg {
	src := e.msg.Meta().Src
	dst := e.msg.Meta().Dst
	var foundSrc, foundDst bool
	var otherMsgs []sim.Msg
	for i, t := range m.inflightEstablishTransactions[src] {
		if t.e == e {
			otherMsgs = t.msgs

			m.inflightEstablishTransactions[src] = append(m.inflightEstablishTransactions[src][:i],
				m.inflightEstablishTransactions[src][i+1:]...)

			foundSrc = true
			break
		}
	}
	for i, t := range m.inflightEstablishTransactions[dst] {
		if t.e == e {
			m.inflightEstablishTransactions[dst] = append(m.inflightEstablishTransactions[dst][:i],
				m.inflightEstablishTransactions[dst][i+1:]...)

			foundDst = true
			break
		}
	}
	if foundSrc && foundDst {
		return otherMsgs
	} else {
		panic("Cannot find the establish event in inflight")
	}
}

// CanSend checks if the network can send a message.
//
//nolint:lll
func (m *OpticalNetworkModel) CanSend(src sim.Port) bool {
	return true
}

// Send sends a message.
func (m *OpticalNetworkModel) Send(msg sim.Msg) *sim.SendError {
	now := m.CurrentTime()
	src := msg.Meta().Src.Name()
	dst := msg.Meta().Dst.Name()
	fmt.Println(src, dst)
	path := m.getPassbyLinksOrChannels(src, dst, "channel")
	channelPath := m.logicalChannels[path[0]][path[1]]
	numHops := len(path) - 1
	fmt.Println("path", path)
	fmt.Println("numHops", numHops)
	latency := channelPath.Latency * sim.VTimeInSec(float64(numHops)) //* 1000
	// latency := channelPath.Latency
	bytePerSecond := channelPath.BytePerSecond
	if path[0] != "RemotePort" {
		m.totalHops += numHops
	}

	if len(path) > 0 {
		transferTime := sim.VTimeInSec(float64(msg.Meta().TrafficBytes) /
			bytePerSecond)
		m.Schedule(transferUpdateEvent{
			time:    now + latency + transferTime,
			handler: m,
			msg:     msg,
		})
	} else {
		panic("No path found, check the topology building process")
		//check what happens here
		// inflightets := m.inflightEstablishTransactions[msg.Meta().Src]
		// for _, t := range inflightets {
		// 	if t.contains(msg.Meta().Dst) {
		// 		t.msgs = append(t.msgs, msg)
		// 		return nil
		// 	}
		// }
	}

	return nil
}

func (m *OpticalNetworkModel) initLink(src sim.Port, dst sim.Port) *Link {
	link := &Link{
		BytePerSecond: 64 * 1e9,
		Latency:       20 * 1e-9,
		Left:          src,
		Right:         dst,
	}
	return link
}
