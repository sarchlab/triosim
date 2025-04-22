// Package networkmodel provides a performance model for the network that
// connects devices.
package networkmodel

import (
	"gitlab.com/akita/akita/v3/sim"
)

// Transfer represents a transfer of data between devices.
type Transfer struct {
	ID       string
	Msg      sim.Msg
	ByteSize uint64
	Src, Dst sim.Port
}

// A NetworkModel can briefly model the behavior of a network.
type NetworkModel interface {
	StartTransfer(t *Transfer) error
	GetCompletedTransfers(now sim.VTimeInSec) []*Transfer
}

// A Switch is a switch in the network.
type Switch struct{}

// A Link is a link in the network that connects two ports.
type Link struct {
	BytePerSecond float64
	Latency       sim.VTimeInSec
	Left, Right   sim.Port
}
