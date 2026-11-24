# Plan: mapping `hyperparameters.yml` to DSENT

## Goal

[hyperparameters.yml](benchmark/gemm_2dmesh/hyperparameters.yml) is the unified interface for simulation parameters. `update_hyperparameters.py` must translate it into DSENT inputs for three models: the mesh router, the electronic link and the point-to-point photonic link.

## How DSENT is driven

DSENT is the gem5 copy at [dependencies/gem5/ext/dsent](dependencies/gem5/ext/dsent). It has no library API. It is a command-line tool that reads three plain-text inputs:

```
dsent -cfg link.cfg [-overwrite "Key1=val; Key2=val;"]
```

1. **A `.cfg` file** of `Key = Value` lines:
   - `ModelName` picks the component: `Router`, `RepeatedLink` (electronic link) or `SWSRLink` (photonic link).
   - The other keys are that model's parameters, such as `NumberBits`.
   - `EvaluateString` lists the results to print, for example `$(Energy>>SWSRLink:Send)` or `$(NddPower>>SWSRLink:Laser)`. `NddPower` is power that doesn't depend on traffic (idle power).
2. **An electrical technology file** (`ElectricalTechModelFilename`): `Bulk45LVT`, `Bulk32LVT`, `Bulk22LVT` or `TG11LVT`.
3. **A photonic technology file** (`PhotonicTechModelFilename`): [Photonics.model](dependencies/gem5/ext/dsent/tech/tech_models/Photonics.model), holding the device constants.

The `.cfg` and `-overwrite` can't change technology-file values. So each run needs a generated `.cfg` per model plus a patched copy of `Photonics.model`.

DSENT outputs energy per event, idle power and area. It doesn't output latency in cycles: that comes from popnet.

## Mapping from `hyperparameters.yml`

| YAML key | DSENT target | Where | Notes |
|---|---|---|---|
| `technology-node` | `ElectricalTechModelFilename` | cfg | `TG-11` → `TG11LVT.model`. **`SOI-45` has no model**; the closest is `Bulk45LVT`. |
| `clock.electrical-network` | `CoreDataRate` (SWSRLink), `Frequency` (Router) | cfg | Hz |
| `topology.flit-size` | `NumberBits` (both links), `NumberBitsPerFlit` (Router) | cfg | |
| `router.virtual-channel-count` | `NumberVirtualChannelsPerVirtualNetwork` | cfg | with `NumberVirtualNetworks = 1` |
| `router.virtual-channel-depth` | `NumberBuffersPerVirtualChannel` | cfg | |
| *(none)* | `NumberInputPorts` / `NumberOutputPorts` | cfg | 5 for a mesh router (4 directions + local) |
| `electronic-link.latency` | `Delay` (RepeatedLink) | cfg | seconds = cycles / clock; a target used to size repeaters |
| *(missing)* | `WireLength` (RepeatedLink), `Length` (SWSRLink) | cfg | metres; see open decision 1 |
| `laser.wavelength-count` | `LinkDataRate` | cfg | bit/s, derived; see "Derived values and checks" |
| `electronic-photonic-interface.ring-tuning-model` | `RingTuningMethod` | cfg | `full-thermal` → `FullThermal`, `bit-reshuffled` → `ThermalWithBitReshuffle`, `electrically-assisted` → `ElectricalAssistWithBitReshuffle` |
| `electronic-photonic-interface.modulator-loss` / `modulator-extinction` | `InsertionLoss` / `ExtinctionRatio` | cfg | dB. Needs `OptimizeLoss = false`; when it is `true`, DSENT picks its own values ([OpticalGraph.cc:90](dependencies/gem5/ext/dsent/model/optical_graph/OpticalGraph.cc#L90)). |
| `laser.efficiency` | `Laser->CW->Efficiency` | photonic tech file | |
| `photonic-link.waveguide-loss` | `Waveguide->LossPerMeter` | photonic tech file | dB/cm × 100 |
| `photonic-link.bit-error-rate` | `SenseAmp->BER` | photonic tech file | |
| `electronic-photonic-interface.ring-drop-loss` / `ring-through-loss` | `Ring->DropLoss` / `Ring->ThroughLoss` | photonic tech file | dB |
| `electronic-photonic-interface.ring-heating-efficiency` | `Ring->HeatingEfficiency` | photonic tech file | K/mW × 1000 = K/W |
| `electronic-photonic-interface.photodetector-capacitance` | `Photodetector->Cap` | photonic tech file | fF × 1e-15 |
| `photonic-link.coupler-loss` | `Coupler->Loss` | photonic tech file | **No effect**: no model reads this key. |

Keys that don't go to DSENT:

| YAML keys | Used by |
|---|---|
| `router.head-latency`, `router.body-latency`, `photonic-link.latency` | popnet only |
| `electronic-link.capacity`, `photonic-link.capacity-per-wavelength` | capacity checks (below) |
| `router.area` | nothing; see open decision 2 |
| `memory-block`, `asic`, `clock.cpu`, `clock.asic`, `workload`, `topology.mesh-dimension`, `topology.cpu-position` | out of DSENT's scope |

## Derived values and checks

Every link, electronic or photonic, carries one flit per clock cycle (popnet assumes the same). DSENT doesn't take a wavelength count: it computes `NumberBits × CoreDataRate / LinkDataRate`, rounded down ([SWSRLink.cc:81](dependencies/gem5/ext/dsent/model/optical/SWSRLink.cc#L81)). So `wavelength-count` is the input and the per-wavelength rate is derived from it.

The two capacity keys are physical upper limits. The simulator checks the achieved rates against them and flags a violation.

```python
required_link_rate = flit_size * clock_hz                    # bit/s, e.g. 25.6e9
rate_per_wavelength = required_link_rate / wavelength_count  # e.g. 6.4e9 → DSENT LinkDataRate

if required_link_rate > electronic_capacity:
    # flag: electronic link over capacity
if rate_per_wavelength > capacity_per_wavelength:
    # flag: photonic wavelength over capacity
```

With the current values, 25.6 Gb/s < 256 Gb/s and 6.4 Gb/s < 32 Gb/s, so both checks pass.

Two things to watch:

1. **Units.** The YAML stores MHz and Gb/s. Convert to Hz and bit/s before computing or comparing.
2. **Rounding.** A floating-point `LinkDataRate` can make DSENT's division give `3.9999…`, which it rounds down to 3 wavelengths. Write it as `int(round(rate_per_wavelength))`.

## Open decisions

1. **Link lengths.** Both link models need a length, and the YAML has none. Either add `electronic-link.length` and `photonic-link.length`, or derive the length from `asic.area` and the mesh pitch.
2. **`router.area`.** DSENT computes router area itself from `flit-size`, the VC count and depth, the port count and `technology-node` (it sums standard-cell areas, so it underestimates the real area). Either remove the key, or keep it as a value to check DSENT's result against.
3. **Capacity violations.** Raise an error and stop, or log a warning and continue. A warning allows exploring over-limit designs on purpose.
4. **`SOI-45`.** Map it to `Bulk45LVT`, or remove it from `supported-values`.
5. **`photonic-link.coupler-loss`.** It has no effect in DSENT. Either remove it, or add it to the loss budget some other way.

## Implementation steps

1. Settle the open decisions and update `hyperparameters.yml`.
2. In `update_hyperparameters.py`, compute the derived values and run the capacity checks.
3. Generate the three `.cfg` files (`Router`, `RepeatedLink`, `SWSRLink`) from the mapping table.
4. Generate the patched copy of `Photonics.model`.
5. Run DSENT on each `.cfg` and parse the energy, idle power and area results.
6. Combine DSENT's per-flit energies with popnet's flit counts.
