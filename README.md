# TrioSim

A lightweight simulator for large-scale DNN workloads on multi-GPU systems. TrioSim supports various parallelism strategies including data parallelism, tensor parallelism, and pipeline parallelism.

## Table of Contents
- [1. Tracer](#1-tracer)
  - [1.1 Trace Collection](#11-trace-collection)
  - [1.2 Trace Data Processing](#12-trace-data-processing)
- [2. TrioSim](#2-triosim)
  - [2.1 Go Installation](#21-go-installation)
  - [2.2 Configuration](#22-configuration)
  - [2.3 Running the Simulator](#23-running-the-simulator)

## 1. Tracer
TrioSim provides processed sample traces for immediate use. These traces are located in the `./sample_trace/` directory.
To start a test quickly: skip the trace collection steps (Section 1); go directly to Section 2 to begin simulation.

### 1.1 Trace Collection

#### Environment Used
- Python: 3.10.12
- CUDA: 12.1
- torch: 2.1.0+cu121
- torchvision: 0.16.0+cu121
- torchaudio: 2.1.0+cu121

#### Dataset
The codes use the ILSVRC2012_img_val dataset. For a quick start, a subset of 256 images is included under ./tracer/data.
#### Usage
To collect traces from PyTorch models, we use PyTorch Profiler to gather layer or operator time information, and the Execution Graph Observer tool to collect detailed input, output, and other tensor or data information. 
The batch size is set via command-line arguments. You can also customize the number of iterations (num_iters) and the model to trace (listmodel) directly in the code.

Here is a code example for collecting trace when batch size is 16:

```bash
python tracer/datacollect.py 16
```

This will generate two types of files:
- `profiler_xx.json`: Contains timing information for each operator
- `graph_xx.json`: Contains detailed tensor information.

For a quick start, generated traces are available in:
- `tracer/data/graph/graph_xx.json`
- `tracer/data/profiler/profiler_xx.json`

### 1.2 Trace Data Processing
The TARGET_OP_PREFIXES variable allows users to define which layers are included in tensor parallelism. By default, it targets 'convolution', 'linear', and 'embedding' layers.

Run the following command to convert the collected traces into TrioSim format:
```bash
python tracer/dataprocess.py
```

The processed traces: tensor.csv and trace.csv, will be available under:
```
./tracer/data/middledata/trace/XXmodel
```

## 2. TrioSim

### 2.1 Go Installation
Install Go by following the official installation guide:
[Go Installation Guide](https://go.dev/doc/install)

### 2.2 Configuration
The simulator can be configured using the following command-line flags:

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `-trace-dir` | string | "../sample_trace/trace2-h100-bs128/vgg13/" | Directory containing trace files |
| `-batch-size` | int | 128 | Original trace batch size |
| `-batch-size-sim` | int | -1 | Simulation batch size (defaults to batch-size) |
| `-bandwidth` | float | 696 | GPU to remote memory bandwidth (GBps) |
| `-ptp-bandwidth` | float | 65 | GPU to GPU bandwidth (GBps) |
| `-GPUnumber` | int | 8 | Number of GPUs |
| `-micro-batch-size` | int | -1 | Micro batch size for pipeline parallelism |
| `-case` | int | 0 | Simulation mode: 0=training, 1=standard data parallel, 2=distributed data parallel, 3=tensor parallel, 4=pipeline parallel |
| `-capacity` | int | 40 | Memory capacity of each device (1 << capacity) |
| `-numCols` | int | -1 | Number of columns in optical network mesh |
| `-numRows` | int | 1 | Number of rows in optical network mesh |
| `-interconnects` | int | 0 | Interconnect type: 0=electrical, 1=optical |

### 2.3 Running the Simulator

#### Basic Usage
1. Navigate to the triosim directory:
```bash
cd triosim
```

2. Run the simulator with your desired configuration:
```bash
go run main.go \
  -batch-size 128 \
  --batch-size-sim 128 \
  -trace-dir ../sample_trace/trace2-h100-bs128/vgg13 \
  --GPUnumber 4 \
  --case 1
```



