# gemm_2dmesh

GEMM `C = A·B` on a 3×3 photonic mesh. A CPU at the centre (Sniper) sends one row
block of `A` and one column block of `B` to each of eight ASICs (native processes)
and gets one tile of `C` back from each. Popnet times the network; Python computes
area, power and energy afterwards from DSENT-derived costs.

```
0 --- 1 --- 2        node id = x + 3y
|     |     |        CPU  : node 4
3 --- 4 --- 5        ASIC : every other node
|     |     |
6 --- 7 --- 8
```

## Run

You need the `chiplet` container running (`docker start chiplet`), the conda env
`lego` (`numpy`, `python-dotenv`, `loguru`, `matplotlib`), and a `.env` in the repo
root that defines `PROJECT_ROOT` and `PYUTILS_RDIR`.

From the repo root, on the host:

```bash
bash benchmark/gemm_2dmesh/run.sh      # ~3 min; prints run_outputs/gemm_2dmesh.XXXXXX
conda activate lego
python benchmark/gemm_2dmesh/scripts/report_gen.py  run_outputs/gemm_2dmesh.XXXXXX
python benchmark/gemm_2dmesh/scripts/plot_report.py run_outputs/gemm_2dmesh.XXXXXX
```

This leaves `report.json` and `figures/*.png` in the run directory.

`report_gen.py` refuses to report if `matC.bin` is wrong, or if the run directory was
reused by a second simulation. Always pass the fresh directory that `run.sh` prints.

## What you get

`report.json`:

| Key | Metric |
|---|---|
| `comm_window` | CPU's first send to its last receive-finish (cycles) |
| `latency` | CPU's last send-finish to its last receive-finish (cycles). `coordinator`: when the CPU finished receiving. `arrival`: when the tile reached the CPU node. |
| `cuts` | For each straight cut of the mesh: capacity, measured traffic and utilisation (bit/cycle; all flits, both directions) |
| `asics` | Per ASIC: hops, flits in/out, cycles its tile waited for the CPU |
| `area` | 24 link directions × (length × pitch + interface) + 9 routers (mm²) |
| `energy` | Static and dynamic power (W), energy (J), energy per bit (J/bit), over the comm window |

A flit on an `h`-hop path costs `h` link traversals and `h + 1` router visits. Every
link direction and router draws static power for the whole comm window.

Figures: `timeline.png` (when each message happens), `round_trip.png` (wire, compute
and wait per ASIC), `power.png` (static vs dynamic), `area.png` (by component).

## Change a parameter

| Parameter | Edit | New simulation? |
|---|---|---|
| ASIC latency (cycles) | `gemm_2dmesh.yml`: 2nd argument of all 8 `bin/asic` entries | Yes |
| Link delay (cycles/hop) | `topology/2dmesh.gv`: `edge [weight=…]` | Yes |
| Flit width (bits per link per cycle) | Five places, see below | Yes |
| Matrix sizes `M`, `K`, `N` | `src/comms_protocol.h` **and** the copies at the top of `report_gen.py` | Yes |
| CPU cores | `gemm_2dmesh.yml`: `-n` in the `run-sniper` args | Yes |
| Link/router energy, power, area | `link_characteristics.json` | No, re-run the two Python scripts |

**Flit width** must agree in five places. For a flit of `8·f` bytes:

| File | Setting |
|---|---|
| `run.sh` | `-f f` |
| `gemm_2dmesh.yml` | Popnet `-F f` |
| `src/comms_protocol.h` | `FLIT_SIZE = 8·f` (bytes) |
| `report_gen.py` | `FLIT_BITS = 64·f` |
| `link_characteristics.json` | `flit_bits = 64·f` |

Only `f` = 1, 2 or 4 works without editing `MsgHeader`, because the 32-byte header
must be a whole number of flits. The wavelength count and bit rate in
`link_characteristics.json` are documentation only; nothing reads them.

Other limits:

- `M` must divide by `ROW_BLK_CNT` and `N` by `COL_BLK_CNT`, and the tile grid must
  have 8 tiles. The build fails otherwise.
- Hop counts in `report_gen.py` assume this 3×3 mesh with the CPU at node 4. A
  different shape or CPU position needs code changes, not just a new `.gv`.
- In `src/cpu.cpp`, a receive issued before its ASIC has both inputs deadlocks the
  simulation. Keep each receiver waiting for the main thread's signal.

## Sweeps

For each design point: edit, `run.sh`, then both Python scripts. Compare the
`report.json` files.

- **Save the costs yourself.** A run directory keeps its YAML, topology,
  `comms_protocol.h` and binaries, but not `link_characteristics.json`.
  `report_gen.py` always reads the current one, so re-running it on an old run
  overwrites that run's `report.json` with today's costs.
- **Check the figure text.** `plot_report.py` hard-codes some wording and one axis:
  "about 290k cycles … about 2k" (`timeline`), "only the CPU wake-up varies"
  (`round_trip`), "mostly waveguide routing … 9 routers" (`area`), and the
  0.1–2000 mW axis (`power`). Edit them if your results no longer match.

## Files

| File | Role |
|---|---|
| `src/cpu.cpp`, `src/asic.cpp` | CPU program (Sniper) and ASIC model (native) |
| `src/comms_protocol.h` | Sizes, tiling, ASIC placement, message header |
| `gemm_2dmesh.yml` | Processes to launch, and Popnet arguments |
| `topology/2dmesh.gv` | Mesh graph and link delay |
| `link_characteristics.json` | Link and router costs, from `run_outputs/dsent_link_costs/mesh_link_assumptions_dsent.json`. Router values are illustrative. |
| `asic_characteristics.json` | Empty: ASIC energy and area are not modelled |
| `run.sh`, `scripts/report_gen.py`, `scripts/plot_report.py` | Simulate, report, plot |
