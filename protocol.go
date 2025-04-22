package triosim

import "gitlab.com/akita/akita/v3/sim"

// A TensorMsg represents the transfer of a tensor package.
type TensorMsg struct {
	sim.MsgMeta
	TensorPkg     []Tensor
	DstRegionName string
	GPUID         int
	Purpose       string
	RoundID       int
}

// Meta returns the meta data of the message.
func (m *TensorMsg) Meta() *sim.MsgMeta {
	return &m.MsgMeta
}
