# Where `update_hyperparameters()` writes

## Context

Each hyperparameter is hard-coded in up to five files. This maps every key in
`hyperparameters.yml` to its consumers. Paths are relative to
`benchmark/gemm_2dmesh/`.

`cpu.cpp`, `asic.cpp` and `makefile` need no edits: they read everything from
`comms_protocol.h` or command-line arguments.

## Fix first: the YAML does not parse

In `supported_units`, prefix each `[name, scale]` pair with `- ` and delete the
stray `[ ]`.

## Keys with a place to write

| Key | Where |
|---|---|
| `clock.clock_rate` | `link_characteristics.json`: `simulated_clock_hz`. Not the `clock_rate:` fields in `gemm_2dmesh.yml`; those are simulator ratios and stay 1. |
| `topology.flit_size` | With `f = bits / 64`: `run.sh` `-f`; `gemm_2dmesh.yml` Popnet `-F`; `comms_protocol.h` `FLIT_SIZE` (bytes); `report_gen.py` `FLIT_BITS`; `link_characteristics.json` `flit_bits`. Only `f` = 1, 2, 4 work. |
| `topology.mesh_dimension` | `comms_protocol.h` `MESH_X_DIM`, `MESH_Y_DIM`, `ASIC_SLOTS`; `topology/2dmesh.gv` edges; `gemm_2dmesh.yml` `bin/asic` entries, Sniper `-n`, Popnet `-A` (= X·Y); `run.sh` `-w` (= X); `report_gen.py` `MESH_X_DIM`, `MESH_Y_DIM`. |
| `topology.cpu_position` | `comms_protocol.h` `CPU_SLOTS` and the gap in `ASIC_SLOTS`; `report_gen.py` `CPU_SLOT`. |
| `workload.matrices_dimensions` | `comms_protocol.h` `M`, `K`, `N`; `report_gen.py` `M`, `K`, `N`. |
| `router.area` | `link_characteristics.json`: `router.area_mm2_per_router`. |
| `router.virtual_channel_count` | `gemm_2dmesh.yml` Popnet `-V`. |
| `router.virtual_channel_depth` | `gemm_2dmesh.yml` Popnet `-B`. |
| `router.head_latency` | `topology/2dmesh.gv`: `node [pipeline_stage_delay=…]`. |
| `electronic_link.latency` / `photonic_link.latency` | `topology/2dmesh.gv`: `edge [weight=…]`. Only one can apply. |
| `laser.wavelength_count` | `link_characteristics.json`: `optical.wavelengths_per_direction`. Nothing reads it. |
| `photonic_link.capacity_per_wavelength` | `link_characteristics.json`: `optical.bit_rate_per_wavelength_bps`. Nothing reads it. |

## Keys to validate, not write

Popnet moves one flit per link per cycle, so link capacity must equal
`flit_size × clock_rate`.

| Key | Current values |
|---|---|
| `electronic_link.capacity` | 128 bit × 2 GHz = 256 Gb/s. Consistent. |
| `photonic_link.capacity_per_wavelength × laser.wavelength_count` | 4 × 32 = 128 Gb/s, which means 64-bit flits. **Conflicts with `flit_size: 128`.** |

## Keys nothing reads yet

Each needs a model before the function can do more than record it.

| Key | What is missing |
|---|---|
| `technology_node`, `laser.efficiency`, `photonic_link.{coupler_loss, waveguide_loss, bit_error_rate}`, `electronic_photonic_interface.*` | A DSENT run or link-budget model that produces the three `optical.*` energy, power and area values in `link_characteristics.json`. |
| `asic.area` | A field in `asic_characteristics.json` (empty) and a term in `report_gen.py` `calc_area`. |
| `memory_block.*` | A memory model in `asic.cpp` and an energy term in `report_gen.py`. |
| `router.body_latency` | The `.gv` has one delay per router. |

## Missing from the YAML

- ASIC compute latency: 2nd argument of each `bin/asic` entry in
  `gemm_2dmesh.yml` (now `1820`).
- `ROW_BLK_CNT`, `COL_BLK_CNT` in `comms_protocol.h`. Their product must equal
  X·Y − 1, or a new mesh size fails to compile.
- Popnet `-O` (output buffer size).

## Decisions for you

1. **Link type.** Add a key that selects electronic or photonic. It picks the edge
   weight and the capacity check.
2. **`photonic_link.latency: 0`.** Gives `weight=0`. Untested in Popnet; try it on
   a throwaway run.
3. **`report_gen.py`.** Have it read `hyperparameters.yml` instead of rewriting
   its constants. Its `hop_count` and `mesh_cuts` assume a 3×3 mesh with the CPU
   at node 4.
4. **`run.sh`.** Also copy `hyperparameters.yml` and `link_characteristics.json`
   into the run directory.

Unverified: that `pipeline_stage_delay` delays only head flits. I read the call
sites in Popnet's `sim_router.cc`, not the whole pipeline.

## Order

1. Fix the YAML; write the loader and unit conversion.
2. Scalars: `M`/`K`/`N`, flit size, clock, router area, `-V`, `-B`, edge weight.
3. Validation: capacity, divisibility, `f`, tile count.
4. Mesh size and CPU position.

## Verification

- With current values, `git diff` is empty after running the function.
- Change one key; `git diff` touches only the files in its row.
- `run.sh`, then `report_gen.py`: it refuses on a wrong `matC.bin` or a
  `flit_bits` mismatch.
