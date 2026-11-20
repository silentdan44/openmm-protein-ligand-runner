# OpenMM protein–ligand runner

Dockerized molecular-dynamics workflow for a protein–ligand complex. It can prepare a membrane, water, and ions; run OpenMM dynamics; and produce a trajectory-analysis report with optional MMPBSA results.

## Reference systems

| Inputs | Use in this repository |
| --- | --- |
| AQP3 + ligand | Smaller end-to-end membrane validation system. AQP3 is a multipass membrane aquaglyceroporin, so use the membrane workflow for a physically representative setup. [UniProt: AQP3](https://www.uniprot.org/uniprotkb/A0A2R8Y2R4/entry) |
| TRPV4 + capsaicin | Larger channel–ligand reference case for optional, more demanding membrane validation. It is not needed for the AQP3 example. |

The AQP3 command below uses a short production segment but still runs the full fixed 1.5 ns equilibration. It checks preparation, membrane construction, dynamics, and analysis; its few picoseconds of production are not suitable for scientific conclusions.

## Build and compute platform

Build the single image once:

```bash
docker build -t openmm-protein-ligand-runner:latest .
```

The image includes OpenMM's CPU, OpenCL, and CUDA support. It can be built without an NVIDIA GPU. With `--platform auto`, the runner tries CUDA, then OpenCL, then CPU. For CUDA, the host needs a supported NVIDIA driver and NVIDIA Container Toolkit; pass the GPU through with `--gpus all`. Without GPU passthrough, automatic selection falls back to available host platforms.

TmDet runs locally in the container. It downloads the wwPDB Chemical Component Dictionary to `/work/data/ccd`; allow network access and writable container storage. With `docker run --rm`, this download is repeated for each new container unless that directory is persisted.

## Example: AQP3 membrane run

```bash
mkdir -p "$HOME/openmm-protein-ligand-results"
docker run --rm \
  -v "$HOME/openmm-protein-ligand-results:/results" \
  openmm-protein-ligand-runner:latest \
  --receptor /app/tests/fixtures/aqp3_receptor.pdb \
  --ligand /app/tests/fixtures/aqp3_ligand.sdf \
  --output /results --platform auto \
  --membrane --orientation tmdet --lipid-type POPC \
  --membrane-padding-nm 1.0 --ionic-strength-molar 0.15 \
  --prod-steps 2000 \
  --minimize-iterations 500 --report-interval 200 \
  --xtc-interval 200 --checkpoint-interval 1000 \
  --analysis-stride 1 \
  --distance-pair A:50 A:180 --distance-pair A:60 A:160
```

`Modeller.addMembrane()` builds both lipid leaflets and adds water and ions; do not add solvent separately. Membrane mode uses Amber ff14SB, Lipid17, TIP3P, and OpenFF ligand parameters. The first version supports one lipid type per bilayer: `POPC`, `POPE`, `DLPC`, `DLPE`, `DMPC`, `DOPC`, or `DPPC`.

## Equilibration protocol

The runner selects a fixed equilibration schedule from `--membrane`. Both schedules minimize with positional restraints, run two 250 ps NVT stages and four 250 ps NPT stages, then continue into production without restraints. Equilibration durations are fixed in physical time and converted to steps using `--timestep-fs`; there are no NVT/NPT step-count flags. Only production length is configured with `--prod-steps`.

Restraint constants are in kJ mol⁻¹ nm⁻². Protein groups contain heavy atoms only: backbone (`N`, `CA`, `C`, `O`, `OXT`) and side chain. All ligands share a schedule on their heavy atoms.

| Stage | Protein–ligand in water: BB / SC / ligand | Water + membrane: BB / SC / ligand / lipid P-Z |
| --- | ---: | ---: |
| Minimization | 1000 / 1000 / 1000 | 1000 / 1000 / 1000 / 1000 |
| NVT 1 | 1000 / 1000 / 1000 | 1000 / 1000 / 1000 / 1000 |
| NVT 2 | 500 / 500 / 500 | 500 / 500 / 500 / 500 |
| NPT 1 | 500 / 250 / 500 | 500 / 250 / 500 / 500 |
| NPT 2 | 250 / 100 / 250 | 250 / 100 / 250 / 250 |
| NPT 3 | 100 / 50 / 100 | 100 / 50 / 100 / 100 |
| NPT 4 and production | 0 / 0 / 0 | 0 / 0 / 0 / 0 |

`Lipid P-Z` restrains lipid phosphorus atoms only along the membrane-normal Z axis. Membrane NPT uses the membrane barostat with isotropic XY scaling, free Z, and zero surface tension. The schedule is a starting protocol; inspect the membrane geometry and simulation observables and extend equilibration when they continue to drift.

For a larger TRPV4 run, replace the two fixture paths with `trpv4_receptor.pdb` and `trpv4_ligand.sdf`. TRPV4 is much larger and can require substantially more memory and runtime, especially for membrane construction on CPU.

## Inputs and membrane orientation

Mount custom receptor and ligand files read-only, then pass their container paths with `--receptor` and `--ligand`. Repeat `--ligand` once per molecule to simulate a shared protein complex containing several ligands. Each SDF must contain exactly one molecule with a 3D conformer in the receptor's coordinate frame. All ligands are parameterized and receive the same staged restraint schedule.

Ligands are labelled `LIG1`, `LIG2`, … in the order the `--ligand` flags were given, and every ligand is analyzed separately: the report gets one RMSD/contacts/hydrogen-bond plot per ligand plus a comparison plot, `metrics.json` gets a per-ligand summary, and `time_series.csv` gets `ligand_rmsd_nm:LIGn`, `ligand_rg_nm:LIGn`, `ligand_contacts:LIGn`, and `protein_ligand_hbonds:LIGn` columns. The contact and hydrogen-bond tables also carry a `ligand_residue` column. `analysis/index.ndx` still groups all ligands as `LIG` for GROMACS tools.

```bash
--receptor /inputs/receptor.pdb \
--ligand /inputs/ligand_a.sdf \
--ligand /inputs/ligand_b.sdf
```

- `--orientation tmdet` runs local TmDet and applies its rigid-body transform to both receptor and ligand, preserving their relative pose.
- `--orientation preoriented` uses the supplied coordinates. The membrane plane must be XY and its center must match `--membrane-center-z-nm`.
- `--orientation manual` rotates the supplied membrane normal onto Z and moves a point on the membrane center plane to `(0, 0, membrane-center-z)`. For example: `--orientation manual --manual-membrane-normal 0 0 1 --manual-membrane-center-angstrom 88 88 106`.

The runner validates input files and simulation counts before starting. Configure temperature, pressure, timestep, random seed, run lengths, output intervals, and platform with the corresponding CLI flags. `--platform auto` probes usable OpenMM contexts in CUDA → OpenCL → CPU order; a named platform must be available. OpenMM selects the platform used internally by `Modeller.addMembrane()` independently of the MD platform.

## Results, validation, and restart

The output directory contains `system_final.pdb`, `trajectory.xtc`, `state_data.csv`, `run_state.json`, `run_environment.json`, checkpoints, and `final_state.xml` after completion. `run_environment.json` records the selected OpenMM platform and its properties, Python/OpenMM versions, OS and CPU details, thread-related environment variables, and NVIDIA GPU/driver details when `nvidia-smi` is available. Resumed sessions are recorded separately. Analysis writes `analysis/report.html`, `analysis/metrics.json`, `analysis/time_series.csv`, plots, and available ligand-contact, hydrogen-bond, and residue-distance tables.

The report includes simulation QC, protein stability and RMSF, ligand pose and pocket contacts, contact and hydrogen-bond occupancy, selected Cα–Cα residue distances, and membrane metrics when applicable. Ligand RMSD is reported against two references: the first production frame (dashed curves in the per-ligand plots) and the input pose from `system_final.pdb`; protein RMSD is always against the first production frame, not the input structure. Optional metrics are omitted with a note when their data is unavailable.

### Plots in the HTML report

| Plot | What it shows | When it is included |
| --- | --- | --- |
| Simulation QC | Temperature and potential energy over time; density and box volume during NPT. | When state samples are available. |
| Protein stability | Whole-protein backbone RMSD; membrane-core Cα RMSD for membrane systems. | When trajectory metrics are available; membrane-core panel only for membrane systems. |
| Protein RMSF | Per-residue Cα RMSF, with separate traces for protein chains. | When per-residue fluctuations can be calculated. |
| Ligand pocket | Ligand RMSD after protein alignment, pocket-contact count, and protein–ligand hydrogen-bond count (all ligands together). | When trajectory metrics are available. |
| Per-ligand pocket | One figure per ligand (`ligand_pocket_LIGn.png`) with that ligand's RMSD against the first production frame and against the input pose (dashed), its own contact and hydrogen-bond counts. | When trajectory metrics are available; one per LIG residue. |
| MMPBSA membrane planes | Phosphorus z distribution, the median lower/upper planes used for `pb_mem` thickness, and rejected outlier phosphates. | `pb_mem` with an automatically estimated membrane thickness. |
| MMPBSA entropy | Running ΔH, −TΔS (interaction entropy) and ΔG, plus the per-frame ΔEEL distribution. | When MMPBSA frames are available. |
| Ligand comparison | RMSD and contact count of every ligand on shared axes. | When the run has more than one ligand. |
| Contact occupancy | Up to 20 protein residues ranked by fraction of frames in contact with the ligand. | When ligand contacts are detected. |
| Hydrogen-bond occupancy | Up to 20 protein–ligand hydrogen bonds ranked by fraction of frames. | When hydrogen bonds are detected. |
| Membrane integrity | Area per lipid, phosphate-plane separation, and lipid counts in the two leaflets. | For membrane runs with measurable lipid geometry. |
| Residue distances | Selected Cα–Cα distances over time, with distributions and mean, median, and standard deviation. | For each run with one or more `--distance-pair` options. |
| MMPBSA energy terms | Per-frame binding-energy components, distributions, and summary statistics. | When MMPBSA results are available. |
| MMPBSA energy summary | Mean and frame standard deviation for the MMPBSA energy components. | When MMPBSA results are available. |
| MMPBSA residue decomposition | Largest per-residue energy contributions. | Only when `--mmpbsa-decomposition` was requested and decomposition data are available. |

An [example AQP3 report](examples/aqp3-report/report.html) is included. It shows the membrane metrics, selected residue distances, and MMPBSA plots together.

Residue distances use `--distance-pair CHAIN:RESID CHAIN:RESID`; repeat the flag for more pairs. Chain IDs and residue numbers refer to `system_final.pdb`; use `_` for a blank chain. Example: `--distance-pair A:550 A:600`.

The membrane validator checks a completed run's status, protein, ligand, lipids, water coverage, both leaflets, ligand–protein proximity, and trajectory:

```bash
docker run --rm --entrypoint python \
  -v "$HOME/openmm-protein-ligand-results:/results:ro" \
  openmm-protein-ligand-runner:latest \
  -m prot_lig_runner.validate_membrane /results
```

This is a structural smoke check. Inspect `system_final.pdb`, `membrane_geometry.json`, and the trajectory before production use; the validator does not establish equilibration or force-field validity.

For long runs, keep the results directory and add `--resume` with identical inputs and settings. The runner stores a configuration fingerprint and periodically writes phase checkpoints. At every phase transition it also serializes the exact `System` in use (`system-nvt.xml`, `system-npt.xml`, `system-prod.xml`) plus the `Integrator` (`integrator.xml`) next to the topology (`system_final.pdb`). On resume these artifacts are loaded directly and the simulation continues from the checkpoint — the chemistry is never rebuilt, so the checkpoint always matches the system (rebuilding solvent is not bit-reproducible because Modeller does not seed water/ion placement, and a rebuilt system can have a different particle count). The source-file hashes in the fingerprint are provenance only: they are ignored when deciding whether a resume is allowed, so a restart after a runner fix still continues from the serialized `System` and `Integrator` instead of being refused.

A restart is seamless. Frames and rows written after the last checkpoint are cut off before the resumed run appends to them: `state_data.csv` is trimmed by step, and `trajectory.xtc` is truncated at a complete-frame boundary. XTC frames are matched by their integer step rather than by time (XTC stores time as a float32, which at 500 ns production lengths is quantized to about 0.03 ps), and the file is scanned header-by-header rather than decoded and rewritten, so a multi-gigabyte trajectory is trimmed in seconds without a second copy on disk. A frame left half-written by a killed process is discarded. Frames are stamped with the absolute simulation step, including when the trajectory file is first created after a mid-run restart, so the state log and trajectory stay aligned row for row. Restarted output therefore continues with the same step numbering, one frame per `--xtc-interval`, and no duplicate or out-of-order frames.

The serialized system files are large (tens of MB per ensemble for membrane systems); delete them only if you will never resume. Runs started before these artifacts existed fall back to rebuilding from inputs, which can fail with a particle-count mismatch. Checkpoints require the same OpenMM version and platform. For Compose, set `RESULTS_DIR` to a host directory outside the project; the default is `../openmm-protein-ligand-runner-results`.

## MMPBSA

Pass `--mmpbsa` during simulation to run single-trajectory gmx_MMPBSA after MD. Do not combine it with `--skip-gromacs-export`; the calculation needs the protein–ligand topology. Add these flags to a production command to analyze the final 20 ns and sample every 10th saved frame:

```text
--mmpbsa --mmpbsa-model gb --mmpbsa-last-ns 20 --mmpbsa-interval 10
```

gmx_MMPBSA estimates one binding affinity per calculation, so a run with several ligands needs `--mmpbsa-ligand`. It accepts the 1-based `--ligand` order (`2`), the label (`LIG2`), or the residue label printed in the report (`A:123:LIG`). With several ligands and no selection the run stops with the list of available ligands; with exactly one ligand the selection is inferred.

```text
--mmpbsa --mmpbsa-ligand 2
```

Omit `--mmpbsa-last-ns` to use all production frames. If the requested window exceeds the available trajectory, all available production frames are used and a warning is logged. `--mmpbsa-interval` samples frames for MMPBSA; `--analysis-stride` independently controls frames loaded for trajectory metrics and plots. Choose `gb`, `pb`, or (for a membrane run) `pb_mem`; `pb_mem` derives the membrane thickness from the phosphate planes unless you pass `--mmpbsa-membrane-thickness-angstrom`. The membrane dielectric defaults to 7.0.

MMPBSA treats water and lipids implicitly while using the full membrane MD trajectory to extract protein and ligand atoms. Results include `analysis/mmpbsa/FINAL_RESULTS_MMPBSA.dat`, `.csv`, `entropy.json`, and `gmx_MMPBSA.log`. The report gives ΔH (MM/PBSA enthalpy) and ΔG = ΔH − TΔS. `--mmpbsa-entropy` selects the entropy estimate:

- `ie` (default): interaction entropy, −TΔS = kT ln⟨exp(ΔE/kT)⟩ − ⟨ΔE⟩, using only the fluctuation of the protein–ligand interaction energy.
- `c2`: variance approximation, −TΔS = σ²(ΔE)/(2RT); cheaper and less sensitive to rare high-energy frames but ignores the shape of the distribution.
- `both`: report both estimates; ΔG is taken from IE. They are alternatives, not additive.
- `nmode`: normal-mode (harmonic) entropy from gmx_MMPBSA's `&nmode` block: ΔH plus Δ(−TΔS) of the complex−receptor−ligand difference, with its SEM. This minimises and diagonalises every frame, so it is only practical for small systems (thousands of atoms, a few frames); on a 27k-atom complex it will not finish in reasonable time.
- `none`: enthalpy only.

The values are computed by gmx_MMPBSA itself (`interaction_entropy` / `c2_entropy` in `&general`) and parsed from its `*_normal_<model>_IE.dat` and `*_normal_<model>_c2_entropy.dat`, so the reported entropy is the upstream implementation, including its block SEM. The local fallback formulas are used only when those files are missing (results from older versions); they match gmx_MMPBSA except that gmx_MMPBSA uses the gas-phase interaction energy (EEL + VDW) while the fallback uses EEL alone, which changes the number in the second decimal.

The mean is subtracted in IE because it is already in ΔH. Neither method is a full normal-mode or quasi-harmonic entropy, and both need tens of frames to converge; with a handful of frames the entropy term is a diagnostic, not a number to quote. Normal-mode entropy is not offered here because minimising every frame of a 27k-atom complex is impractical.

For `pb_mem`, the membrane thickness is estimated from the lipid phosphate planes of the built system unless `--mmpbsa-membrane-thickness-angstrom` is given. Per-leaflet medians with a MAD cut are used, so flipped phosphates do not move the plane; `analysis/mmpbsa/membrane_planes.json` records the planes, counts and rejected atoms, and `mmpbsa_membrane_planes.png` shows the selection. The membrane dielectric defaults to 7.0 (`--mmpbsa-membrane-dielectric`).

This is an approximate end-state estimate, not an absolute binding free energy. Use it to rank ligands under one protocol, ideally across independent replicas; frame-to-frame variation is descriptive because adjacent frames are correlated. Per-residue decomposition is optional via `--mmpbsa-decomposition` and costs additional computation.

### Atom types and memory on large complexes

Interchange writes one atom type per atom for SMIRNOFF systems. ParmEd, which gmx_MMPBSA uses for the GROMACS→Amber conversion, then sets `NTYPES = NATOM`, and the quadratic `NONBONDED_PARM_INDEX` produced an 18 GB `analysis/mmpbsa/COM.prmtop` for a 27 340-atom pentamer; sander was OOM-killed at ~112 GB RSS. `export_mmpbsa_complex` now merges types that share element, particle type, sigma and epsilon (27 340 → 20 types on that complex, prmtop 12.6 MB), and sander completes in a few minutes. If you prepare a complex elsewhere and see a prmtop of hundreds of megabytes, check `%FLAG POINTERS`: the second number is `NTYPES` and must be small, not equal to `NATOM`.

To rebuild analysis from a completed run, or run MMPBSA when its topology files are present:

```bash
docker run --rm --entrypoint python \
  -v "$HOME/openmm-protein-ligand-results:/results" \
  openmm-protein-ligand-runner:latest \
  -m prot_lig_runner.analysis --output /results --mmpbsa
```

### Preliminary report during a run

While the simulation container is running, start a second container with the same results directory mounted and request a snapshot:

```bash
docker run --rm --entrypoint python \
  -v "$HOME/openmm-protein-ligand-results:/results" \
  openmm-protein-ligand-runner:latest \
  -m prot_lig_runner.analysis --output /results --preview --analysis-stride 10
```

Open `analysis_preview/report.html`. Repeat the command to refresh it; the final `analysis/` report is separate. The preview copies the data available at invocation and discards any incomplete last trajectory frame. QC plots can include equilibration, while trajectory metrics use production frames only, so they appear after production has started and frames have been saved. The report marks its phase and preliminary status; conclusions can change as sampling grows. `--distance-pair` can override the pairs saved with the run. MMPBSA is not run in preview mode. For a consistent snapshot, do not resume or restart the simulation while the preview is copying files.

## Diagnostics and tests

On failure, `failure_diagnostics.json` records the phase, last step and metrics, likely explanation, suggested next step, and any close atom pairs. During dynamics, the runner checks energy and kinetic temperature for non-finite values, extreme temperature, or large energy jumps. These checks identify warning signs; they do not prove a cause.

Run the unit tests in Docker:

```bash
docker run --rm -w /work -e PYTHONPATH=/work/src \
  -v "$PWD:/work:ro" --entrypoint python \
  openmm-protein-ligand-runner:latest \
  -m unittest discover -s tests -v
```
