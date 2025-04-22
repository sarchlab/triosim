package networkmodel

import (
	"fmt"
	"math"
	"regexp"
	"strconv"

	"gitlab.com/akita/akita/v3/sim"
)

// A transferUpdateEvent is an event that is scheduled when a transfer is
// likely to be completed.
type transferUpdateEvent struct {
	time    sim.VTimeInSec
	handler sim.Handler
	msg     sim.Msg
}

func (e transferUpdateEvent) Time() sim.VTimeInSec {
	return e.time
}

func (e transferUpdateEvent) Handler() sim.Handler {
	return e.handler
}

func (e transferUpdateEvent) IsSecondary() bool {
	return false
}

type NetworkNodes struct {
	nodetype  string
	nodeports sim.Port
}

type Route struct {
	src          sim.Port
	dst          sim.Port
	links        []*PSLink
	bw           float64
	msg          sim.Msg
	progress     float64
	timeLeft     float64
	updateTime   sim.VTimeInSec
	scheduleTime sim.VTimeInSec
}

type PSLink struct {
	link   *Link
	routes map[string]*Route
}

// Graph data structure
type Graph map[string]map[string]float64

// A PacketSwitchingNetworkModel is a simple network model that can estimate
// the arrival time of transfers.
type PacketSwitchingNetworkModel struct {
	sim.HookableBase
	sim.EventScheduler
	sim.TimeTeller

	busyNodes       map[string]bool
	pendingDelivery map[string][]sim.Msg

	networkNodes map[string]*NetworkNodes
	links        map[string][]*PSLink
	routes       map[string]*Route
}

// NewPacketSwitchingNetworkModel creates a new PacketSwitchingNetworkModel.
func NewPacketSwitchingNetworkModel(
	es sim.EventScheduler,
	tt sim.TimeTeller,
) *PacketSwitchingNetworkModel {
	m := &PacketSwitchingNetworkModel{
		EventScheduler:  es,
		TimeTeller:      tt,
		busyNodes:       make(map[string]bool),
		pendingDelivery: make(map[string][]sim.Msg),
		networkNodes:    make(map[string]*NetworkNodes),
		links:           make(map[string][]*PSLink),
		routes:          make(map[string]*Route),
	}

	return m
}

// PlugIn plugs a port into the network.
func (m *PacketSwitchingNetworkModel) PlugIn(port sim.Port, bufSize int) {
	port.SetConnection(m)
}

func (m *PacketSwitchingNetworkModel) PlugInWithDetails(port sim.Port, bufSize int, nodetype string) {
	node := NetworkNodes{nodetype: nodetype, nodeports: port}
	m.networkNodes[port.Name()] = &node //switch node and end node should have different names.
	m.PlugIn(port, bufSize)
}

// Unplug removes a port from the network.
func (m *PacketSwitchingNetworkModel) Unplug(port sim.Port) {
	delete(m.networkNodes, port.Name()) //switch node and end node should have different names.
}

// NotifyAvailable notifies the network that the port is available to send messages.
func (m *PacketSwitchingNetworkModel) NotifyAvailable(
	now sim.VTimeInSec,
	port sim.Port,
) {
	pendingDelivery := m.pendingDelivery[port.Name()]

	for len(pendingDelivery) > 0 {
		msg := pendingDelivery[0]
		err := port.Recv(msg)
		if err != nil {
			break
		}

		pendingDelivery = pendingDelivery[1:]
	}

	m.pendingDelivery[port.Name()] = pendingDelivery

	if len(pendingDelivery) == 0 {
		delete(m.busyNodes, port.Name())
	}
}

// AddLink adds a link between two ports.
func (m *PacketSwitchingNetworkModel) AddLink(
	left, right sim.Port,
	bytePerSecond float64,
	latency sim.VTimeInSec,
) {
	m.links[left.Name()] = append(m.links[left.Name()], &PSLink{
		link: &Link{
			Left:          left,
			Right:         right,
			BytePerSecond: bytePerSecond,
			Latency:       latency,
		},
		routes: make(map[string]*Route),
	})
	// can delete for Hop backup, shouldn't be deleted for other cases
	m.links[right.Name()] = append(m.links[right.Name()], &PSLink{
		link: &Link{
			Left:          right,
			Right:         left,
			BytePerSecond: bytePerSecond,
			Latency:       latency,
		},
		routes: make(map[string]*Route),
	})
}

// Handle checks if the transfers are completed.
func (m *PacketSwitchingNetworkModel) Handle(e sim.Event) error {
	switch e := e.(type) {
	case transferUpdateEvent:
		return m.handleTransferUpdateEvent(e)
	default:
		panic("unknown event type")
	}
}

// only schedule NextHappenEvent case
func (m *PacketSwitchingNetworkModel) handleTransferUpdateEvent(
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

	if !m.checkScheduleEvent(e) {
		m.schedulehNextHappenEvent()
		return nil
	}

	e.msg.Meta().RecvTime = m.CurrentTime()
	err := msg.Meta().Dst.Recv(msg)

	if err != nil {
		m.busyNodes[msg.Meta().Dst.Name()] = true
		m.pendingDelivery[msg.Meta().Dst.Name()] = append(
			m.pendingDelivery[msg.Meta().Dst.Name()],
			msg,
		)
	} else {
		routeDelete := m.UpdateRoute(msg)
		m.UpdateProgressNextHappenEvent(routeDelete)
	}

	return nil
}

// CanSend checks if the network can send a message.
func (m *PacketSwitchingNetworkModel) CanSend(src sim.Port) bool {
	return true
}

// Send and only schedule NextHappenEvent
func (m *PacketSwitchingNetworkModel) Send(msg sim.Msg) *sim.SendError {
	route := m.findRoute(msg)
	m.UpdateProgressNextHappenEvent(route)
	m.schedulehNextHappenEvent()
	return nil
}

func (m *PacketSwitchingNetworkModel) checkScheduleEvent(e transferUpdateEvent) bool {
	route, found := m.routes[e.msg.Meta().ID]
	if !found || route == nil {
		// fmt.Println("Route not found for message ID:", e.msg.Meta().ID)
		return false
	}

	if route.scheduleTime != e.time {
		return false
	}
	return true
}

func (m *PacketSwitchingNetworkModel) UpdateProgressNextHappenEvent(route *Route) {
	msgCount := 0
	var routesUpdate = make(map[string]*Route)
	for _, psLink := range route.links {
		for key, routeUpdate := range psLink.routes {
			routesUpdate[key] = routeUpdate
		}
	}

	for _, routeUpdate := range routesUpdate {
		minBW := math.Inf(1)
		for _, psLink := range routeUpdate.links {
			msgCount = len(psLink.routes)
			bw := psLink.link.BytePerSecond
			//for HOP test random slowdown
			// if routeUpdate.src.Name() == "GPU0Port" {
			// 	randnum := rand.Float64()
			// 	if randnum < 4.0/16.0 {
			// 		randnum = (randnum + 0.5) * 20
			// 		bw = bw / randnum
			// 	}
			// }
			currentBW := bw / float64(msgCount)
			if minBW > currentBW {
				minBW = currentBW
			}
			fmt.Println("bwlink", float64(msgCount), psLink.link.BytePerSecond, minBW)
		}
		updateTime := routeUpdate.updateTime
		timeSinceSend := float64(m.CurrentTime() - updateTime)
		progress := routeUpdate.progress + timeSinceSend*routeUpdate.bw // bw is 0 at first without set
		otherMsgBytesLeft := float64(routeUpdate.msg.Meta().TrafficBytes) - progress
		var timeLeft float64
		if otherMsgBytesLeft >= 0 {
			timeLeft = otherMsgBytesLeft / minBW
		} else {
			timeLeft = 0.0
			// fmt.Println(float64(otherMsgBytesLeft)/minBW, routeUpdate.msg.Meta().ID)
		}
		routeUpdate.bw = minBW
		routeUpdate.progress = progress
		routeUpdate.timeLeft = timeLeft
		now := m.CurrentTime()
		routeUpdate.updateTime = now
		routeUpdate.scheduleTime = now + sim.VTimeInSec(timeLeft)
		fmt.Println("time for this msg data,", sim.VTimeInSec(timeLeft), minBW, otherMsgBytesLeft)
	}
}

// ScheduleEvent schedules all the events which update.
func (m *PacketSwitchingNetworkModel) schedulehNextHappenEvent() {
	ScheduleTime := sim.VTimeInSec(math.Inf(1))
	routeSchedule := Route{}
	for _, route := range m.routes {
		if ScheduleTime > route.scheduleTime {
			ScheduleTime = route.scheduleTime
			routeSchedule = *route
		}
	}
	if ScheduleTime == sim.VTimeInSec(math.Inf(1)) {
		return
	}
	// fmt.Println(m.CurrentTime()*1000000, "schedule msg id is ",
	// 	routeSchedule.msg.Meta().ID, ", schedule time is ", routeSchedule.scheduleTime)
	m.Schedule(transferUpdateEvent{
		time:    routeSchedule.scheduleTime,
		handler: m,
		msg:     routeSchedule.msg,
	})
}

func (m *PacketSwitchingNetworkModel) UpdateRoute(msg sim.Msg) *Route {
	routeDelete := m.routes[msg.Meta().ID]
	delete(m.routes, msg.Meta().ID)

	for _, psLink := range routeDelete.links {
		delete(psLink.routes, msg.Meta().ID)
	}
	return routeDelete
}

func extractNumber(s string) (int, error) {
	// Define the regex pattern to match "gpu" followed by a number.
	re := regexp.MustCompile(`GPU(\d+)Port`)
	// Find the first match.
	match := re.FindStringSubmatch(s)
	// Check if there was a match.
	if len(match) < 2 {
		return 0, fmt.Errorf("no number found in the input string")
	}
	// Convert the matched number part to an integer.
	number, err := strconv.Atoi(match[1])
	if err != nil {
		return 0, fmt.Errorf("error converting string to number: %v", err)
	}

	return number, nil
}

func (m *PacketSwitchingNetworkModel) FindNeighbor(node string, aim string) []int {
	neighbors := []int{}
	switch aim {
	case "in":
		neighbors = m.findInNeighbors(node)
	case "out":
		neighbors = m.findOutNeighbors(node)
	default:
		// Handle unexpected 'aim' values if necessary
	}
	neighbors = deleteDuplicate(neighbors)
	return neighbors
}

func (m *PacketSwitchingNetworkModel) findInNeighbors(node string) []int {
	neighbors := []int{}
	for left, PSLink := range m.links {
		for _, PSLink := range PSLink {
			if PSLink.link.Right.Name() == node+"Port" {
				if left == "RemotePort" {
					continue
				}
				id, err := extractNumber(left)
				if err != nil {
					fmt.Println(err)
				} else {
					neighbors = append(neighbors, id)
				}
			}
		}
	}
	return neighbors
}

func (m *PacketSwitchingNetworkModel) findOutNeighbors(node string) []int {
	neighbors := []int{}
	for _, PSLink := range m.links[node+"Port"] {
		if PSLink.link.Right.Name() == "RemotePort" {
			continue
		}
		id, err := extractNumber(PSLink.link.Right.Name())
		if err == nil {
			neighbors = append(neighbors, id)
		} else {
			fmt.Println(err)
		}
	}
	return neighbors
}

// deleteDuplicate removes duplicate integers from a slice.
func deleteDuplicate(nums []int) []int {
	seen := make(map[int]bool)
	result := []int{}
	for _, num := range nums {
		if !seen[num] {
			seen[num] = true
			result = append(result, num)
		}
	}
	return result
}
func (m *PacketSwitchingNetworkModel) findLinkFromPorts(src sim.Port, dst sim.Port) *PSLink {
	for _, PSLink := range m.links[src.Name()] {
		if PSLink.link.Right.Name() == dst.Name() {
			return PSLink
		}
	}
	for _, PSLink := range m.links[dst.Name()] {
		if PSLink.link.Right.Name() == src.Name() {
			return PSLink
		}
	}
	panic("link not found")
}

func (m *PacketSwitchingNetworkModel) findRoute(msg sim.Msg) *Route {
	if route, found := m.routes[msg.Meta().ID]; found {
		return route
	}

	src := msg.Meta().Src.Name()
	dst := msg.Meta().Dst.Name()
	graph := m.initializeGraph()
	path, ok := m.calculateShortestPath(graph, src, dst)

	if !ok {
		panic("no path found")
	}

	// Convert path from string slice to []*Link
	PSLinks := make([]*PSLink, 0, len(path))

	for i := range path[:len(path)-1] {
		nodePortSrc := m.networkNodes[path[i]].nodeports
		nodePortDst := m.networkNodes[path[i+1]].nodeports
		PSLinkFind := m.findLinkFromPorts(nodePortSrc, nodePortDst)
		PSLinks = append(PSLinks, PSLinkFind)
	}

	newRoute := &Route{
		src:   msg.Meta().Src,
		dst:   msg.Meta().Dst,
		links: PSLinks,
		msg:   msg,
	}

	for _, PSLink := range PSLinks {
		PSLink.routes[msg.Meta().ID] = newRoute
	}

	m.routes[msg.Meta().ID] = newRoute
	return newRoute
}

func (m *PacketSwitchingNetworkModel) initializeGraph() Graph {
	graph := make(Graph)

	// For each link in the network, add an entry in the graph for both directions
	for src, psLinks := range m.links {
		if _, ok := graph[src]; !ok {
			graph[src] = make(map[string]float64)
		}

		for _, psLink := range psLinks {
			dst := psLink.link.Right.Name()

			// Initialize the inner map for the destination node if not already present
			if _, ok := graph[dst]; !ok {
				graph[dst] = make(map[string]float64)
			}

			weight := psLink.link.BytePerSecond

			graph[src][dst] = weight
			graph[dst][src] = weight
		}
	}

	return graph
}

func (m *PacketSwitchingNetworkModel) initializeDistances(
	graph Graph,
	src string,
) (map[string]float64, map[string]string) {
	dist := make(map[string]float64)
	prev := make(map[string]string)

	for node := range graph {
		if node == src {
			dist[node] = 0
		} else {
			dist[node] = math.Inf(1)
		}
	}

	return dist, prev
}

func (m *PacketSwitchingNetworkModel) calculateShortestPath(graph Graph, src, dst string) ([]string, bool) {
	dist, prev := m.initializeDistances(graph, src)

	for len(dist) > 0 {
		minNode := ""
		minDist := math.Inf(1)
		for node := range dist {
			if d := dist[node]; d < minDist {
				minDist = d
				minNode = node
			}
		}

		if minNode == "" || minNode == dst {
			break
		}

		// Update distances to neighbors
		for neighbor, weight := range graph[minNode] {
			if alt := dist[minNode] + weight; alt < dist[neighbor] {
				dist[neighbor] = alt
				prev[neighbor] = minNode
			}
		}

		// This node has been visited
		delete(dist, minNode)
	}

	return m.reconstructPath(prev, dst)
}

func (m *PacketSwitchingNetworkModel) reconstructPath(prev map[string]string, dst string) ([]string, bool) {
	path := []string{}
	for node := dst; node != ""; node = prev[node] {
		path = append([]string{node}, path...)
	}

	if len(path) == 0 {
		return nil, false
	}

	return path, true
}
