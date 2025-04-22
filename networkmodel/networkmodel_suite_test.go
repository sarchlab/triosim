package networkmodel

import (
	"testing"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

//go:generate mockgen -write_package_comment=false -package=$GOPACKAGE -destination=mock_sim_test.go gitlab.com/akita/akita/v3/sim EventScheduler,TimeTeller,Port

func TestNetworkModel(t *testing.T) {
	RegisterFailHandler(Fail)
	RunSpecs(t, "Network Model Suite")
}
