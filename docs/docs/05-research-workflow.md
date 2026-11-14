# From a first simulation to electrical/photonic NoC experiments

Start with `benchmark/matmul_test`. Its YAML launches three GPGPU-Sim processes
and one Sniper process in phase 1, then Popnet in phase 2. `interchiplet` repeats
the phases, feeding network delays from the previous round into the next
workload execution. The source entry point is `interchiplet/srcs/interchiplet.cpp`.

## Follow one communication

In `benchmark/matmul_test/matmul.cpp`, the CPU sends two 80,000-byte arrays to
each GPU, receives three 800-byte arrays, and adds the results. There are nine
application-level transfers and 482,400 payload bytes in total.

The GPU kernel in `matmul.cu` launches one block of 10 by 10 threads. Each thread
computes one entry from the top-left 10 by 10 part of the matrix product, with
an inner dimension of 100. Every GPU receives the same inputs and performs the
same computation. The original `matmul.yml` uses a CPU binary without a
numerical check. The `matmul_photonic_mesh.yml` variant builds a separate CPU
binary with `VALIDATE_MATMUL` enabled. It checks all 100 reduced output values
against a CPU calculation of three times that 10 by 10 tile, then prints
`MATMUL_CHECK PASS` with a checksum. This validates the computation the example
actually performs; the three GPUs still do redundant work.

`#ifdef VALIDATE_MATMUL` in C++ selects code at compile time. The `checked`
Makefile target passes `-DVALIDATE_MATMUL` and writes a separate executable,
`bin/matmul_checked_c`. Think of it as building a second version of the same
Python program with the validation branch enabled. The reference calculation
and its logging run inside Sniper and therefore contribute to the checked
workload's completion cycles. Compare architectures using the same checked
binary and workload configuration.

The CPU API in `interchiplet/srcs/apis_c.cpp` invokes simulator-specific syscalls.
Running that CPU executable directly on the host does not reproduce its
communication behavior inside Sniper.

In Python terms:

```python
# Conceptual application flow, not an executable Python API.
for gpu in [(0, 1), (1, 0), (1, 1)]:
    send(gpu, A)
for gpu in [(0, 1), (1, 0), (1, 1)]:
    send(gpu, B)
results = [receive(gpu) for gpu in [(0, 1), (1, 0), (1, 1)]]
C = sum(results)
```

The coordinator records transfers in `bench.txt`. The active writer is
`NetworkBenchList::dumpBench` in `interchiplet/includes/net_bench.h`; it emits:

```text
source_cycle destination_cycle source_node destination_node transfer_size descriptor
```

Coordinates are flattened as `node = x + y * width`. With `-w 6`, the GPU node
IDs are 6, 1, and 7. The transfer-size field is calculated from payload size and
flit configuration and includes a header; it is **not a byte count**. For this
example with `-f 2`, the large transfers are encoded as 5001 and small ones as
51. Do not infer application payload bytes from this field without accounting
for headers and padding.

## Run without mixing outputs from different experiments

The existing `chiplet` container mounts the repository at `/workspace/sim`.
Run this from the host when that container is running:

```bash
docker exec chiplet bash -c '
set -e
cd /workspace/sim
source setup_env.sh
mkdir -p run_outputs
run_dir=$(mktemp -d /workspace/sim/run_outputs/matmul_baseline.XXXXXX)
chmod 755 "$run_dir"
cp benchmark/matmul_test/matmul.yml "$run_dir/input.original.yml"
printf "Output directory: %s\n" "$run_dir"
cd "$run_dir"
/workspace/sim/interchiplet/bin/interchiplet \
    /workspace/sim/benchmark/matmul_test/matmul.yml -w 6 -f 2 \
    > coordinator.log 2>&1
'
```

The YAML remains at its original path because `$BENCHMARK_ROOT` is resolved
from the YAML's directory. The copied YAML records the input; executing that
copy without adjusting paths would change where benchmark binaries are found.
Use a fresh output directory for every run: `delayInfo.txt` from a previous run
would otherwise be loaded at the start.

`proc_r3_p1_t2` means round 3, phase 1, process/thread index 2. In this YAML that
index is the GPU at `(1, 1)`. These names are output identities, not simulated
chiplet coordinates.

## Export and plot results with Python

Use the host path corresponding to the printed container output directory:

```bash
python3 tools/analyze_run.py run_outputs/matmul_baseline.REPLACE \
    --output /tmp/legosim-analysis
```

This writes `summary.json` and `rounds.csv`. With Matplotlib installed in the
chosen Python environment, add `--plot` to produce `convergence.png` and
`convergence.svg`. Choose an output directory you can write to; container-created
directories may belong to root.

The reader rejects missing child termination records, nonzero child statuses,
truncated logs, and inconsistent final cycle summaries. It labels a completed
run without a convergence message separately. For the checked mesh variant it
also requires a `MATMUL_CHECK PASS` marker in every phase-1 CPU log and records
the checksums. The original example has no such numerical check. Log validation
does not establish physical model accuracy.

The coordinator's default convergence test is
`abs(current_cycles - previous_cycles) / current_cycles < 0.005`, with a default
limit of five rounds. It tests stability of total completion cycles. It does
not prove convergence of every packet delay or accuracy against hardware.

When convergence is detected, the coordinator stops after phase 1. Thus the
last Popnet report often belongs to the previous round. The reader keeps that
round number instead of presenting it as a new network measurement for the
final round. `bench.txt` is overwritten by each phase 1; `delayInfo.txt` records
the latest completed phase 2. They can represent different rounds at exit.

## Verify what architecture is actually simulated

The current example names `test/mesh_6_6.gv` via `-G`, but omits `-R`. Popnet
defaults to routing mode 0. `CGraphTopo::init()` is selected by graph mode 4,
so the graph file is not used in this run. Its `-A 100 -c 1` arguments configure
100 routers with one-dimensional addresses. The named mesh file is also absent
at that relative path in this checkout.

Before a topology comparison, explicitly select the intended routing mode,
provide an existing graph path, and ensure the router count, contiguous numeric
node IDs, and flattened endpoint IDs agree. A relative graph path is resolved
from the phase-2 process directory. Prefer an absolute path when preparing an
experiment configuration.

## Map the four research metrics to evidence

| Metric | Existing evidence | Work needed for defensible E/P results |
|---|---|---|
| Workload latency | Coordinator completion cycles | Establish physical clock and clock-domain conversions; validate workload |
| Network latency | Popnet average packet completion delay | Separate packet, message, and workload latency; account for E/O and O/E conversion and optical resource contention |
| Throughput | Transfers and timing can support a defined rate | Specify useful bytes, observation interval, offered load, and whether the goal is application throughput or saturation throughput |
| Energy | Popnet reports ORION-based power; DSENT has electrical and photonic cost models | Connect activity counts to appropriate event energy, add idle/leakage/laser/tuning energy, and validate units and technology |
| Area | DSENT exposes active and photonic area | Inventory instantiated components and multiplicities; specify treatment of waveguides, shared lasers, and footprint |

The graph model supports edge delay (`weight`) and energy (`energy_per_hop`),
plus router pipeline delay and forwarding energy. See
`popnet_chiplet/graph_lib/graph.cpp` and
`popnet_chiplet/changes/shortest_path_routing.cpp`. Edge delay reaches
`CGraphTopo::getWireDelay` in `popnet_chiplet/sim_router.cc`.

However, the graph's `energyTable` is computed and stored without a downstream
use in the inspected source. The printed network power instead comes from
ORION calls in `power_template`, aggregated by
`sim_foundation::simulation_results`. Setting graph energy attributes does not
currently establish a photonic energy report.

`gem5/ext/dsent/configs/photonic-link.cfg` provides an `SWSRLink` example with
send energy, leakage, laser and ring-tuning power, and active/photonic area.
This checkout's DSENT CMake target builds a Python 2 extension. The
`tools/dsent_cli.cc` wrapper lets the vendored models run as a standalone
executable inside the `chiplet` container; `tools/dsent_link_costs.py` builds
it and saves machine-readable link costs.
No DSENT coupling was found in the inspected coordinator/Popnet path.

A proposed energy accounting layer would compute:

```text
dynamic energy = sum(event count * energy per event)
static energy  = integral(active static-power components over simulated time)
total energy   = dynamic energy + static energy
```

For an always-on laser and constant tuning/leakage power, the integral reduces
to power times duration. If lasers or tuning resources are shared or gated,
their allocation and active intervals must be represented explicitly. Do not
count a shared laser once per logical link.

A practical next sequence after validating the example's actual computation is
to establish a genuine electrical graph baseline, then specify the optical architecture (point-to-point
WDM links, shared waveguides, or a switched optical network). That choice
determines which bandwidth, arbitration, wavelength, conversion, and static-power
behaviors must be added. Merely assigning a smaller delay to an edge would not
establish those behaviors.

## First optical mesh demonstration

`benchmark/matmul_test/matmul_photonic_mesh.yml` runs the same CPU and three GPU
processes on a 2 by 2 graph. Node IDs are 0, 1, 2, and 3, with coordinates
`(0,0)`, `(1,0)`, `(0,1)`, and `(1,1)` respectively. Run the coordinator with
`-w 2 -f 2`; the configured Popnet process uses `-A 4 -c 1 -R 4` and an actual
graph file at `benchmark/matmul_test/topology/photonic_mesh_2_2.gv`.

Each graph edge stands for an independent full-duplex optical connection. Its
four-cycle weight is a placeholder for electrical-to-optical conversion,
waveguide propagation, optical-to-electrical conversion, and timing margin.
The graph gives Popnet the topology and link delay. The fixed flit size and
router timing still come from Popnet's existing model. Popnet does not model
laser physics, wavelength allocation, or ring resonance in this configuration.

`benchmark/matmul_test/photonic_mesh_assumptions.json` records explicit
placeholder values: 1 GHz simulated clock; 128-bit flits; eight wavelengths at
16 Gb/s each per direction; 10 mm waveguide length; 5 µm waveguide pitch;
1.5 pJ per transmitted bit per hop; 1 mW laser and 0.2 mW ring tuning power
per direction; and 0.02 mm² of transmitter/receiver area per direction. The
four electrical routers use additional placeholders: 0.4 pJ per transmitted
bit per router visit, 0.5 mW static power per router, and 0.04 mm² per router.
These are illustrative inputs rather than characterized technology values.

Use a fresh directory, keeping the YAML in its original location so its
`$BENCHMARK_ROOT` expansion remains correct:

```bash
docker exec chiplet bash -c '
set -e
cd /workspace/sim
source setup_env.sh
make -C benchmark/matmul_test -f makefile checked
run_dir=$(mktemp -d /workspace/sim/run_outputs/photonic_mesh.XXXXXX)
chmod 755 "$run_dir"
cp benchmark/matmul_test/matmul_photonic_mesh.yml "$run_dir/input.photonic.yml"
cp benchmark/matmul_test/photonic_mesh_assumptions.json "$run_dir/photonic_mesh_assumptions.json"
cp benchmark/matmul_test/topology/photonic_mesh_2_2.gv "$run_dir/photonic_mesh_2_2.gv"
printf "Output directory: %s\n" "$run_dir"
cd "$run_dir"
/workspace/sim/interchiplet/bin/interchiplet \
    /workspace/sim/benchmark/matmul_test/matmul_photonic_mesh.yml \
    -w 2 -f 2 > coordinator.log 2>&1
'
```

After the run completes, export both the simulation history and the NoC
energy/area estimate. Replace the directory suffix with the printed one:

```bash
python3 tools/analyze_run.py run_outputs/photonic_mesh.REPLACE \
    --output /tmp/legosim-photonic-analysis
python3 tools/photonic_mesh_report.py run_outputs/photonic_mesh.REPLACE \
    --output /tmp/legosim-photonic-analysis
```

Add `--plot` to either command when Matplotlib is available. The second command
writes `photonic_link_metrics.json`, `photonic_trace.csv`, and a one-row
`metrics.csv` that can be combined with later experiments. With `--plot`, it
also writes an energy/area figure showing optical and router contributions.

The reporter reads the last phase-1 `bench.txt`, counts shortest-path link hops
for every transfer, and multiplies the transmitted bits (including header
flits) by the stated energy per bit per hop. Its throughput is an upper bound
on **encoded payload bytes divided by the full workload duration**; the trace
can include padding and has no exact original byte length. Static laser and
tuning energy assumes every direction remains powered for the full workload.
Area consists of a waveguide routing footprint (`length × pitch`) plus one
transmitter/receiver pair per direction. Router energy counts one visit to the
source router and one to each subsequent router on the route, for every flit.
Router static energy assumes all four routers are powered for the full workload.
The report gives optical-link, router, and combined NoC totals. The printed
Popnet/ORION power report is kept separate, as it comes from a different
electrical model and has not been reconciled with these illustrative values.

To turn this prototype into a research model, calibrate the E/O and O/E delay,
line rate, bit energy, laser/tuning duty cycle, and physical footprint to a
particular process and link design. Add explicit wavelength and receiver
contention if multiple flows share an optical resource. A 2 by 2 graph
demonstrates the data path and accounting; it does not validate device physics.

## Controlled electrical/optical link comparison

The four chiplet coordinates form four undirected graph edges. Each edge can be
electrical or optical. `benchmark/matmul_test/scenarios/` contains explicit maps
for all electrical, all optical, vertical optical, and horizontal optical
layouts. Their common placeholder costs are in
`benchmark/matmul_test/mesh_link_assumptions.json`. Edit a scenario's `edges`
map to try another layout. The edge key `0-1` means the link between nodes 0
and 1 in the graph, not the direction of traffic; Popnet provides both
directions for each edge.

`tools/mesh_study.py` generates a topology and YAML in a fresh `run_outputs`
directory, runs the checked CPU/GPU workload inside the `chiplet` container,
then writes `analysis/mesh_metrics.json` and `analysis/mesh_transfers.csv`.
For example, from the repository root on the host:

```bash
python3 -B tools/mesh_study.py run benchmark/matmul_test/scenarios/all_electrical.json
python3 -B tools/mesh_study.py run benchmark/matmul_test/scenarios/all_optical.json
python3 -B tools/mesh_study.py run benchmark/matmul_test/scenarios/vertical_optical.json
```

Each command prints its new run directory. Pass the three printed directories
to the comparison command:

```bash
/home/xli56/miniconda3/envs/gem5/bin/python -B tools/mesh_study.py compare \
    run_outputs/mesh_all_electrical.REPLACE \
    run_outputs/mesh_all_optical.REPLACE \
    run_outputs/mesh_vertical_optical.REPLACE \
    --output run_outputs/mesh_comparison --plot
```

It writes one-row-per-scenario `comparison.csv`, full `comparison.json`, and a
four-metric figure. The existing `gem5` Conda environment has Matplotlib;
plain system `python3` can run `compare` without `--plot` to write CSV/JSON.

This comparison holds the checked workload, flit width, Popnet router setup,
clock, and assumed 128 Gb/s per-direction bandwidth fixed. The placeholder
edge delay is four cycles for optical and eight for electrical. Electrical
and optical link energy, static power, and area use their own values from the
shared assumptions file; the same electrical-router placeholder applies to
every scenario. The reporter checks that traffic, checksum, checked CPU binary,
and assumptions match before making a comparison. The graph route with the
least modeled delay determines how many of each link type a transfer uses.

The CPU checksum validates the redundant 10 by 10 computation in this example.
The technology values remain illustrative, and Popnet still has no wavelength
contention or photonic device physics. A single run per scenario shows a design
workflow; statistical variation, calibration, and workload diversity are
further research steps.

The generated graph labels each edge with its link type and a numeric delay.
Popnet uses the numeric `weight` for timing; the type label is recorded for the
external cost report. In this experiment both types carry one 128-bit flit per
simulated cycle, so the comparison changes link delay and estimated physical
costs, while keeping link bandwidth equal. To study bandwidth advantages or
shared wavelength contention, the simulator's link service behavior must be
extended and tested separately.

## Replace the link placeholders with DSENT estimates

The vendored DSENT models estimate the costs of a 10 mm electrical
`RepeatedLink` and an optical `SWSRLink` at the same 128-bit, 1 GHz mesh
interface. The optical configuration uses eight wavelengths at 16 Gb/s each,
which gives the same nominal 128 Gb/s as the electrical bus. Run:

```bash
python3 tools/dsent_link_costs.py
```

This compiles `tools/dsent_cli.cc` with the vendored DSENT source in the
`chiplet` container and writes generated DSENT configs, raw output,
`link_costs.json`, and `mesh_link_assumptions_dsent.json` under
`run_outputs/dsent_link_costs/`. The electrical technology is
`Bulk45LVT.model`; the optical model also uses `Photonics.model`. For the
current checkout, DSENT estimates per **directed** link are:

| Model | Send energy | Idle power | Route plus device area |
|---|---:|---:|---:|
| Electrical | 0.393 pJ/bit | 0.0555 mW | 0.410 mm² |
| Optical | 0.125 pJ/bit | 14.97 mW | 0.0465 mm² |

The optical idle value includes about 5.70 mW laser, 7.95 mW ring tuning, and
1.32 mW leakage. DSENT's electrical wire area is the sum across 128 wires;
its optical route area is one waveguide. The generated mesh assumptions split
each model's route area from its electronics while preserving their total.

Use that file explicitly for a scenario:

```bash
python3 -B tools/mesh_study.py run \
    benchmark/matmul_test/scenarios/all_optical.json \
    --assumptions run_outputs/dsent_link_costs/mesh_link_assumptions_dsent.json
```

Run all scenarios with the same assumptions file before using `compare`.
The original `mesh_link_assumptions.json` remains a separate toy baseline.
One checked run per layout in this checkout produced:

| Layout | Packet latency (network cycles) | Useful payload rate (MB/s) | NoC energy (µJ) | NoC area (mm²) |
|---|---:|---:|---:|---:|
| All electrical | 3561.67 | 167.758 | 12.66 | 3.438 |
| All optical | 3360.78 | 167.700 | 354.58 | 0.532 |
| Vertical optical | 3593.22 | 167.681 | 183.64 | 1.985 |

The tiny throughput and completion-time differences are not evidence of a
speedup; this is one workload and one run per layout. In this particular
model, always-on optical laser and ring-tuning power dominate energy, while
the 128 electrical wires dominate route area.

The DSENT costs describe isolated links. The router energy and area, optical
four-cycle edge delay, and workload remain illustrative; the electrical
eight-cycle value is a DSENT *input constraint*, not a measured link latency.
No wavelength contention, laser sharing, thermal variation, or photonic
propagation delay is simulated. The summed route and device areas are resource
estimates, not a floorplan. Use a technology and topology appropriate to your
design before making physical claims.

## Reprice saved traffic when only cost assumptions change

Popnet uses the saved graph's delays to simulate packet timing. Once a run is
validated, changing only an energy, idle-power, or area coefficient does not
change its packet trace. `tools/mesh_reprice.py` recalculates those costs from
the saved trace without rerunning the CPU, GPUs, or Popnet:

```bash
/home/xli56/miniconda3/envs/gem5/bin/python -B tools/mesh_reprice.py \
    run_outputs/mesh_all_electrical.REPLACE \
    run_outputs/mesh_all_optical.REPLACE \
    run_outputs/mesh_vertical_optical.REPLACE \
    --assumptions run_outputs/dsent_link_costs/mesh_link_assumptions_dsent.json \
    --output run_outputs/mesh_reprice_dsent --plot
```

Use the three directories from the **same original comparison**. The command
validates their checked workload, traffic, binaries, fixed YAML settings, and
original assumptions before repricing. It writes `repriced.csv`,
`repriced.json`, an energy/area plot, and the source validation report.
Latency, throughput, and workload cycles are copied from each validated run;
the output marks every row as cost-only repricing.

It refuses changes to the mesh width, flit width, clock, router-stage timing,
link delay, link length, or nominal link rate. Rerun `mesh_study.py run` for
those changes. Repricing a saved run with its original assumptions reproduced
all four reported metrics exactly in this checkout. Applying the DSENT link
costs to the earlier toy-cost traces gave 354.44 µJ for all optical and
183.56 µJ for vertical optical, close to the separate DSENT-costed reruns;
their small difference follows from the runs' slightly different completion
times and hence different integrated idle energy.

For a quick Python sensitivity study, copy the DSENT assumptions JSON and
change `optical.static_power_w_per_direction`, then pass the copy to
`mesh_reprice.py`. At fixed timing and traffic, this example's optical
idle-power coefficient would need to fall to about **0.77%** of the current
DSENT value for the all-optical or vertical-optical layout to match the
all-electrical total energy. This is an algebraic target under a fixed-trace
assumption; it does not model laser gating, wake-up cost, or altered timing.

## Expand the photonic mesh to 3×3

The scenario JSON now defines the square mesh width and every edge's link
type. `tools/mesh_study.py` generates the DOT graph, Popnet router count, and
process coordinates from it. The coordinator maps chiplet position `(x, y)`
to Popnet node `x + y × width`; for a 3×3 mesh, the IDs are:

```text
6──7──8   GPU at 6 and 8
│  │  │
3──4──5
│  │  │
0──1──2   CPU at 0, GPU at 2
```

By default, the CPU and three GPUs are placed at the four corners. The CPU program accepts
six optional GPU coordinates; its original two-argument form keeps the 2×2
placement. Popnet receives `-A 9 -c 1` and the coordinator receives `-w 3`.
The central and intermediate routers are included in traffic routing and in
idle power and area accounting, even when they host no workload process.

Run the three layouts with the same DSENT assumptions file:

```bash
python3 -B tools/mesh_study.py run \
    benchmark/matmul_test/scenarios/all_electrical_3x3.json \
    --assumptions run_outputs/dsent_link_costs/mesh_link_assumptions_dsent.json
python3 -B tools/mesh_study.py run \
    benchmark/matmul_test/scenarios/all_optical_3x3.json \
    --assumptions run_outputs/dsent_link_costs/mesh_link_assumptions_dsent.json
python3 -B tools/mesh_study.py run \
    benchmark/matmul_test/scenarios/vertical_optical_3x3.json \
    --assumptions run_outputs/dsent_link_costs/mesh_link_assumptions_dsent.json
```

Use the printed run directories with `mesh_study.py compare` to get CSV,
JSON, and a four-metric plot. The checker requires identical workload traffic,
numerical checksums, binaries, link assumptions, and fixed simulator settings.
It also confirms that graph mode used nine routers and the trace exercised all
four configured endpoint nodes. Compare link layouts *within* one mesh width; changing the
mesh width also changes endpoint placement, router/link count, and path length.

One checked run per 3×3 layout produced:

| Layout | Packet latency (network cycles) | Useful payload rate (MB/s) | NoC energy (µJ) | NoC area (mm²) |
|---|---:|---:|---:|---:|
| All electrical | 4349.11 | 167.323 | 26.52 | 10.195 |
| All optical | 3371.89 | 167.531 | 1054.70 | 1.477 |
| Vertical optical | 4027.56 | 167.430 | 540.92 | 5.836 |

This table has nine routers and 24 directed links, compared with four routers
and eight directed links in the 2×2 mesh. The fixed, always-on link power
assumption therefore makes optical energy rise strongly with mesh size. The
single-run latency and throughput values need repeated experiments before
claiming a performance trend.

For another square mesh or a custom E/P pattern, generate an editable scenario
in Python:

```bash
python3 tools/generate_mesh_scenario.py \
    --width 4 --layout vertical-optical --set 0-1=optical \
    --output benchmark/matmul_test/scenarios/custom_4x4.json
```

The generator supports `all-optical`, `all-electrical`, `vertical-optical`,
and `horizontal-optical`; repeat `--set EDGE=KIND` for individual links.
Then pass the JSON to `mesh_study.py run`. The 2×2 and 3×3 sizes have been
run end to end; larger sizes have generated and route-checked graphs but still
need end-to-end validation on this checkout.

The 3×3 experiment still contains one CPU and three GPUs, not nine compute
chiplets. By default, each link type gets one length and cost profile from the
assumptions file. Individual links can now override the complete profile, as
shown below. A real layout still needs measured physical lengths and timing
assumptions that match its topology.

## Include an NPU endpoint

The scenario generator can replace the worker at the far corner with an NPU
process while keeping one CPU and two GPUs. For example:

```bash
python3 tools/generate_mesh_scenario.py \
    --width 3 --layout vertical-optical \
    --worker3-kind npu --npu-compute-cycles 128 \
    --output benchmark/matmul_test/scenarios/my_vertical_3x3_npu.json
python3 -B tools/mesh_study.py run \
    benchmark/matmul_test/scenarios/my_vertical_3x3_npu.json \
    --assumptions run_outputs/dsent_link_costs/mesh_link_assumptions_dsent.json
```

`benchmark/matmul_test/matmul_npu.cpp` is a small data-driven accelerator
wrapper. It receives the same two 100×100 matrices as the GPU workers,
computes the same 10×10 output tile, and sends it back. After the second input
transfer completes at cycle `t`, it schedules its output write at
`t + npu_compute_cycles`. The reporter checks that timing relationship in
every NPU log, the NPU tile checksum, and the CPU's element-by-element result.

The `128` compute cycles above are an explicit placeholder. They do not come
from SCALE-Sim, RTL, or hardware measurement. The NoC energy and area report
includes routers and links; it excludes the NPU core's energy and area. Replace
the compute-cycle value with a measured or separately modeled value for your
accelerator before making performance claims. The link costs and bandwidth
limitations described earlier still apply.

The checked-in scenarios `all_electrical_3x3_npu.json`,
`all_optical_3x3_npu.json`, and `vertical_optical_3x3_npu.json` hold the CPU,
two GPUs, NPU, workload bytes, and 128-cycle NPU delay fixed while changing
the E/P links. One run per layout gave:

| Link layout | Packet latency (network cycles) | Useful payload rate (MB/s) | NoC energy (µJ) | NoC area (mm²) |
|---|---:|---:|---:|---:|
| All electrical | 4395.67 | 167.646 | 26.49 | 10.195 |
| All optical | 3371.89 | 167.679 | 1053.77 | 1.477 |
| Vertical optical | 4027.56 | 167.668 | 540.17 | 5.836 |

For each layout, run `mesh_study.py run` with the same DSENT assumptions file,
then pass the printed directories to `mesh_study.py compare --plot`. Its CSV
and JSON contain the numeric data; each run's `analysis/mesh_metrics.json`
also records the NPU checksum and verifies that output WRITE followed the
last input READ by exactly 128 cycles. The small throughput differences here
are not evidence of an architectural gain from a single run.

### Use a SCALE-Sim cycle estimate for the NPU

For a reproducible timing input, the vendored SCALE-Sim model runs the NPU's
single GEMM tile: `M=10`, `N=10`, `K=100`. The example configuration uses a
16×16 output-stationary array, 64 KiB each of input, weight, and output SRAM,
and SCALE-Sim's calculated interface bandwidth. It reports **129 compute
cycles**, with zero stall cycles for this example. Run the wrapper with a
Python environment containing NumPy and tqdm:

```bash
/home/xli56/miniconda3/envs/gem5/bin/python -B tools/npu_scalesim.py
```

This writes `run_outputs/scalesim_npu_tile/npu_timing.json` plus SCALE-Sim's
`COMPUTE_REPORT.csv`. The manifest records the config, GEMM topology, report,
and vendored SCALE-Sim source hashes. Pass it to the scenario generator:

```bash
python3 -B tools/generate_mesh_scenario.py \
    --width 3 --layout all-optical \
    --name all_optical_3x3_npu_scalesim \
    --worker3-kind npu \
    --npu-timing run_outputs/scalesim_npu_tile/npu_timing.json \
    --output benchmark/matmul_test/scenarios/all_optical_3x3_npu_scalesim.json
python3 -B tools/mesh_study.py run \
    benchmark/matmul_test/scenarios/all_optical_3x3_npu_scalesim.json \
    --assumptions run_outputs/dsent_link_costs/mesh_link_assumptions_dsent.json
```

Use `--layout all-electrical` or `--layout vertical-optical` and a distinct
name/output to compare links with the same workload and accelerator timing.
The run directory saves the timing manifest and copies of the SCALE-Sim config,
GEMM topology, and compute report. The reporter checks their hashes and verifies
that the NPU's output WRITE follows its final input READ by exactly 129 cycles.

The 129 cycles are a SCALE-Sim estimate under these explicit array and memory
assumptions. They do not include an RTL implementation, memory-system stalls
beyond this model, or a measured NPU. NoC energy and area still exclude the NPU
core. Replace the config and topology with your architecture, rerun SCALE-Sim,
and regenerate each scenario before interpreting accelerator performance.

One checked run per layout with this 129-cycle timing input produced the files
in `run_outputs/mesh_comparison_3x3_npu_scalesim/`:

| Link layout | Packet latency (network cycles) | Useful payload rate (MB/s) | NoC energy (µJ) | NoC area (mm²) |
|---|---:|---:|---:|---:|
| All electrical | 4360.11 | 167.579 | 26.50 | 10.195 |
| All optical | 3371.89 | 167.798 | 1053.03 | 1.477 |
| Vertical optical | 4027.67 | 167.668 | 540.16 | 5.836 |

All three runs converged and passed the three-round workload checksum and NPU
timing checks. These single-run values are an example for learning and plotting,
not a statistical comparison or a calibrated physical design result.

## Move compute chiplets on the mesh

The scenario can also place the CPU and each worker at any four distinct mesh
nodes. Coordinates are `X,Y`, with Popnet node ID `X + Y × width`. Omitted
positions keep the default corners. For example, move the CPU to the center of
the 3×3 optical mesh and put its first GPU at `(0,0)`:

```bash
python3 -B tools/generate_mesh_scenario.py \
    --width 3 --layout all-optical \
    --name all_optical_3x3_npu_center_cpu \
    --worker3-kind npu \
    --npu-timing run_outputs/scalesim_npu_tile/npu_timing.json \
    --cpu 1,1 --worker1 0,0 \
    --output benchmark/matmul_test/scenarios/all_optical_3x3_npu_center_cpu.json
python3 -B tools/mesh_study.py run \
    benchmark/matmul_test/scenarios/all_optical_3x3_npu_center_cpu.json \
    --assumptions run_outputs/dsent_link_costs/mesh_link_assumptions_dsent.json
```

The generator also accepts `--worker2` and `--worker3`. It saves a complete
`placements` map in the JSON, and rejects duplicate or out-of-bounds positions.
The run passes CPU coordinates to the GPU and NPU workers; the reporter checks
that every trace transfer joins the placed CPU to a worker and that all four
endpoints were exercised. To compare placements, run a baseline using the same
source-built binaries and assumptions, then pass both run directories to
`mesh_study.py compare --plot`. The comparison checks the same role-level
workload traffic while allowing its network coordinates to differ.

One matched, checked run of each placement produced
`run_outputs/mesh_placement_comparison_3x3_npu/comparison.csv` and a plot:

| Placement | Packet latency (network cycles) | Useful payload rate (MB/s) | NoC energy (µJ) | NoC area (mm²) |
|---|---:|---:|---:|---:|
| Default corners | 3371.89 | 167.735 | 1053.42 | 1.477 |
| CPU `(1,1)`, GPU 1 `(0,0)` | 3366.67 | 167.813 | 1051.58 | 1.477 |

The same 3×3 optical graph gives the same area. The small differences in
latency, throughput, and energy are one-run observations under the illustrative
link model; repeat runs before interpreting them as placement effects.

## Give one link its own physical profile

The scenario's `edges` map still chooses electrical or optical links. An
optional `edge_parameters` map can give any named edge a complete profile:
delay in cycles, dynamic joules per transmitted bit, idle watts per direction,
length in millimetres, routing pitch in micrometres, and interface area in
square millimetres per direction. The other edges use their link-type defaults.

For an example, run DSENT again at 5 mm and choose an **illustrative**
two-cycle Popnet delay for that edge:

```bash
python3 -B tools/dsent_link_costs.py \
    --output run_outputs/dsent_optical_5mm \
    --optical-length-mm 5 --optical-edge-delay-cycles 2
python3 -B tools/generate_mesh_scenario.py \
    --width 3 --layout all-optical \
    --name all_optical_3x3_npu_short_edge01 \
    --worker3-kind npu \
    --npu-timing run_outputs/scalesim_npu_tile/npu_timing.json \
    --edge-profile 0-1=run_outputs/dsent_optical_5mm/mesh_link_assumptions_dsent.json \
    --output benchmark/matmul_test/scenarios/all_optical_3x3_npu_short_edge01.json
python3 -B tools/mesh_study.py run \
    benchmark/matmul_test/scenarios/all_optical_3x3_npu_short_edge01.json \
    --assumptions run_outputs/dsent_link_costs/mesh_link_assumptions_dsent.json
```

DSENT estimates the 5 mm link's send energy, non-data-dependent power, and
area. The two-cycle value is a Popnet timing assumption; it is not a measured
propagation delay. `mesh_study.py` copies the source profile into the run
directory, checks its hash, and writes `analysis/mesh_edges.csv` with each
edge's assumptions, transmitted bits, dynamic energy, idle energy, and area.
If equal-delay paths would incur different modeled link energy, the reporter
refuses to assign one arbitrary path to the packet trace.

With the same CPU/GPU/NPU placement, worker binaries, workload, and base link
assumptions, one run of each optical mesh gave:

| Optical mesh | Packet latency (network cycles) | Useful payload rate (MB/s) | NoC energy (µJ) | NoC area (mm²) |
|---|---:|---:|---:|---:|
| All 10 mm links | 3371.89 | 167.735 | 1053.42 | 1.477 |
| Edge `0–1` at 5 mm, two-cycle delay | 3370.78 | 167.758 | 1050.12 | 1.437 |

The comparison CSV and plot are in
`run_outputs/mesh_edge_profile_comparison_3x3_npu/`. The 0.040 mm² area
reduction follows directly from the shorter bidirectional route. Energy also
falls in this isolated DSENT estimate. Performance differences are small
single-run observations under the illustrative delay and traffic model.

For a cost-only sensitivity run on this 3×3 result, pass its saved
`link_assumptions.json` as the repricing starting point; it records the mesh
width as well as the base link costs. Repricing with that unchanged file
reproduced latency, throughput, energy, and area exactly.

## Interpret the throughput column

The checked matrix workload sends two 100×100 matrices of 64-bit values to
each of three workers and receives one 10×10 result tile from each. That is
`3 × (2 × 10000 × 8 + 100 × 8) = 482400` useful bytes. Every transfer fills
whole 16-byte flit payloads, so the earlier encoded-payload upper bound happens
to equal the exact useful-byte count here. The reporter now checks all nine
packet sizes, directions, and counts before adding `useful_payload_bytes` and
`useful_payload_throughput_MBps` to its JSON and comparison CSV.
The count includes each replicated matrix transfer to the three workers;
it measures bytes carried for this workload, not unique matrix information.

The throughput denominator is the **full coordinator workload time** at the
assumed 1 GHz clock. For the 5 mm edge example, it is
`482400 / (2875579 / 1e9) / 1e6 = 167.758 MB/s`.
This is application communication goodput for this particular matrix run.
A network saturation throughput study requires a separate offered-load sweep;
Popnet's packet-latency column answers a different question.
For these converged runs, useful throughput uses the final workload round
(round 3), while packet latency comes from the last completed Popnet phase
(round 2). Both round IDs are saved in the comparison CSV.

The comparison CSV is ordinary Python input:

```python
import csv

with open("run_outputs/mesh_edge_profile_comparison_3x3_npu/comparison.csv") as file:
    for row in csv.DictReader(file):
        print(row["scenario"], float(row["useful_payload_throughput_MBps"]))
```

## Sweep offered NoC traffic

The checked matrix workload gives one application payload rate. To inspect
how the NoC responds as injection rises, `tools/popnet_offered_load.py` runs
Popnet's plain four-field trace mode directly. It generates a seeded stream
of four-flit packets from every router to another router, then reuses the
same trace for each topology at a given injection probability:

```bash
/home/xli56/miniconda3/envs/gem5/bin/python -B tools/popnet_offered_load.py \
    --scenario \
      benchmark/matmul_test/scenarios/all_electrical_3x3.json \
      benchmark/matmul_test/scenarios/all_optical_3x3.json \
      benchmark/matmul_test/scenarios/vertical_optical_3x3.json \
    --assumptions run_outputs/dsent_link_costs/mesh_link_assumptions_dsent.json \
    --rates 0.02 0.05 0.1 0.2 0.3 0.4 \
    --warmup-cycles 500 --measure-cycles 1500 \
    --packet-flits 4 --seed 17 \
    --output run_outputs/popnet_offered_3x3 --plot
```

The output includes `offered_load.csv`, `offered_load.json`, an offered-versus-
delivered plot, every generated trace, each Popnet log, and packet-completion
records. Rates are normalized as flits per node per cycle and include packet
headers. The measurement window counts injected packets by injection cycle and
delivered packets by tail-flit completion cycle. A packet injected during
warmup may finish inside the window, so the two counts need not match exactly.

For this single seeded sweep, delivered traffic rose with offered traffic and
then flattened near 0.8 flits per node per cycle across the three link layouts.
Mean latency of packets completed in the window rose sharply at high load.
The CSV records packets still undelivered at the simulation horizon; use that
backlog alongside the latency curve. These are finite-window observations for
one uniform-random traffic seed and one packet size, not a validated saturation
threshold. The optical graph changes link delays, while Popnet still gives the
electrical and photonic links the same one-flit-per-cycle capacity and has no
wavelength-contention model. This synthetic sweep does not run the CPU, GPU, or
NPU workers and does not calculate device energy or area.

To see how much the uniform-random traffic draw changes these observations,
repeat the study with several independent seeds:

```bash
/home/xli56/miniconda3/envs/gem5/bin/python -B tools/popnet_offered_load.py \
    --scenario \
      benchmark/matmul_test/scenarios/all_electrical_3x3.json \
      benchmark/matmul_test/scenarios/all_optical_3x3.json \
      benchmark/matmul_test/scenarios/vertical_optical_3x3.json \
    --assumptions run_outputs/dsent_link_costs/mesh_link_assumptions_dsent.json \
    --rates 0.02 0.1 0.2 0.3 0.4 \
    --warmup-cycles 500 --measure-cycles 1500 --packet-flits 4 \
    --seeds 17 23 29 \
    --output run_outputs/popnet_offered_3x3_multiseed --plot
```

Each seed uses the same trace across the three layouts. The raw
`offered_load.csv` has one row per seed, rate, and layout. The
`offered_load_summary.csv` gives the mean and minimum-to-maximum range across
seeds, and the plot shades that range. This range describes variation across
three traffic draws; it is neither a confidence interval nor uncertainty in
the photonic device model. At high offered load, completed-packet latency
excludes packets still queued at the end of the run, so inspect the unfinished
packet count before interpreting that curve.

In the three-seed 3×3 run above, the lowest offered rate averaged 0.082 flits
per node per cycle. Completed-packet mean latency was 28.6 cycles for the
electrical mesh, 20.5 for the all-optical mesh, and 24.4 for the vertical-link
optical mesh. At the highest rate, offered traffic averaged 1.609 flits per
node per cycle, but delivered traffic averaged only 0.779, 0.783, and 0.781,
respectively. Around 3,765–3,788 packets per layout remained unfinished at
the 2,000-cycle horizon. These measurements show a finite-window delivery
plateau and growing backlog, not a steady-state saturation estimate. The
small high-load differences between layouts lie within their three-seed
traffic ranges under this equal-capacity link assumption.
