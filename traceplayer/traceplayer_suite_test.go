package traceplayer

import (
	"testing"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

//go:generate mockgen -write_package_comment=false -package=$GOPACKAGE -destination=mock_sim_test.go gitlab.com/akita/akita/v3/sim EventScheduler,TimeTeller,Port
//go:generate mockgen -write_package_comment=false -package=$GOPACKAGE -destination=mock_timemodel_test.go github.com/syifan/triosim/timemodel TimeEstimator

func TestTraceplayer(t *testing.T) {
	RegisterFailHandler(Fail)
	RunSpecs(t, "Traceplayer Suite")
}
