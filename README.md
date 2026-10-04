# OpenMM protein–ligand runner

Dockerized molecular-dynamics workflow for a protein–ligand complex. It can prepare a membrane, water, and ions; run OpenMM dynamics; and produce a trajectory-analysis report with optional MMPBSA results.

## Reference systems

| Inputs | Use in this repository |
| --- | --- |
| AQP3 + ligand | Smaller end-to-end membrane smoke test. AQP3 is a multipass membrane aquaglyceroporin, so use the membrane workflow for a physically representative setup. [UniProt: AQP3](https://www.uniprot.org/uniprotkb/A0A2R8Y2R4/entry) |
| TRPV4 + capsaicin | Larger channel–ligand reference case for optional, more demanding membrane validation. It is not needed for the quick smoke test. |

The short AQP3 command below checks that preparation, membrane construction, dynamics, and analysis run. Its few picoseconds of production are not suitable for scientific conclusions.

## Build and compute platform

Build the single image once:

```bash
docker build -t openmm-protein-ligand-runner:latest .
```

The image includes OpenMM's CPU, OpenCL, and CUDA support. It can be built without an NVIDIA GPU. With `--platform auto`, the runner tries CUDA, then OpenCL, then CPU. For CUDA, the host needs a supported NVIDIA driver and NVIDIA Container Toolkit; pass the GPU through with `--gpus all`. Without GPU passthrough, automatic selection falls back to available host platforms.

TmDet runs locally in the container. It downloads the wwPDB Chemical Component Dictionary to `/work/data/ccd`; allow network access and writable container storage. With `docker run --rm`, this download is repeated for each new container unless that directory is persisted.

## Quick start: AQP3 membrane smoke test

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
  --nvt-steps 1000 --npt-steps 1000 --prod-steps 2000 \
  --minimize-iterations 500 --report-interval 200 \
  --xtc-interval 200 --checkpoint-interval 1000 \
  --analysis-stride 1 \
  --distance-pair A:50 A:180 --distance-pair A:60 A:160
```

`Modeller.addMembrane()` builds both lipid leaflets and adds water and ions; do not add solvent separately. Membrane mode uses Amber ff14SB, Lipid17, TIP3P, and OpenFF ligand parameters. The first version supports one lipid type per bilayer: `POPC`, `POPE`, `DLPC`, `DLPE`, `DMPC`, `DOPC`, or `DPPC`.

For a larger TRPV4 run, replace the two fixture paths with `trpv4_receptor.pdb` and `trpv4_ligand.sdf`. TRPV4 is much larger and can require substantially more memory and runtime, especially for membrane construction on CPU.

## Inputs and membrane orientation

Mount custom receptor and ligand files read-only, then pass their container paths with `--receptor` and `--ligand`. The receptor must be a PDB; the ligand must be one molecule with a 3D conformer in the receptor's coordinate frame.

- `--orientation tmdet` runs local TmDet and applies its rigid-body transform to both receptor and ligand, preserving their relative pose.
- `--orientation preoriented` uses the supplied coordinates. The membrane plane must be XY and its center must match `--membrane-center-z-nm`.
- `--orientation manual` rotates the supplied membrane normal onto Z and moves a point on the membrane center plane to `(0, 0, membrane-center-z)`. For example: `--orientation manual --manual-membrane-normal 0 0 1 --manual-membrane-center-angstrom 88 88 106`.

The runner validates input files and simulation counts before starting. Configure temperature, pressure, timestep, random seed, run lengths, output intervals, and platform with the corresponding CLI flags. `--platform auto` probes usable OpenMM contexts in CUDA → OpenCL → CPU order; a named platform must be available. OpenMM selects the platform used internally by `Modeller.addMembrane()` independently of the MD platform.

## Results, validation, and restart

The output directory contains `system_final.pdb`, `trajectory.xtc`, `state_data.csv`, `run_state.json`, checkpoints, and `final_state.xml` after completion. Analysis writes `analysis/report.html`, `analysis/metrics.json`, `analysis/time_series.csv`, plots, and available ligand-contact, hydrogen-bond, and residue-distance tables.

The report includes simulation QC, protein stability and RMSF, ligand pose and pocket contacts, contact and hydrogen-bond occupancy, selected Cα–Cα residue distances, and membrane metrics when applicable. Optional metrics are omitted with a note when their data is unavailable.

### Plots in the HTML report

| Plot | What it shows | When it is included |
| --- | --- | --- |
| Simulation QC | Temperature and potential energy over time; density and box volume during NPT. | When state samples are available. |
| Protein stability | Whole-protein backbone RMSD; membrane-core Cα RMSD for membrane systems. | When trajectory metrics are available; membrane-core panel only for membrane systems. |
| Protein RMSF | Per-residue Cα RMSF, with separate traces for protein chains. | When per-residue fluctuations can be calculated. |
| Ligand pocket | Ligand RMSD after protein alignment, pocket-contact count, and protein–ligand hydrogen-bond count. | When trajectory metrics are available. |
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

For long runs, keep the results directory and add `--resume` with identical inputs and settings. The runner stores a configuration fingerprint and periodically writes phase checkpoints. Checkpoints require the same OpenMM version and platform. For Compose, set `RESULTS_DIR` to a host directory outside the project; the default is `../openmm-protein-ligand-runner-results`.

## MMPBSA

Pass `--mmpbsa` during simulation to run single-trajectory gmx_MMPBSA after MD. Do not combine it with `--skip-gromacs-export`; the calculation needs the protein–ligand topology. Add these flags to a production command to analyze the final 20 ns and sample every 10th saved frame:

```text
--mmpbsa --mmpbsa-model gb --mmpbsa-last-ns 20 --mmpbsa-interval 10
```

Omit `--mmpbsa-last-ns` to use all production frames. If the requested window exceeds the available trajectory, all available production frames are used and a warning is logged. `--mmpbsa-interval` samples frames for MMPBSA; `--analysis-stride` independently controls frames loaded for trajectory metrics and plots. Choose `gb`, `pb`, or (for a membrane run) `pb_mem`; `pb_mem` also requires `--mmpbsa-membrane-thickness-angstrom`. The membrane dielectric defaults to 7.0.

MMPBSA treats water and lipids implicitly while using the full membrane MD trajectory to extract protein and ligand atoms. Results include `analysis/mmpbsa/FINAL_RESULTS_MMPBSA.dat`, `.csv`, and `gmx_MMPBSA.log`; the HTML report includes MMPBSA plots when results are available. This is an approximate end-state estimate, not an absolute binding free energy. Frame-to-frame variation is descriptive because adjacent frames are correlated; use equilibrated trajectories and independent replicas for scientific interpretation. Per-residue decomposition is optional via `--mmpbsa-decomposition` and costs additional computation.

To rebuild analysis from a completed run, or run MMPBSA when its topology files are present:

```bash
docker run --rm --entrypoint python \
  -v "$HOME/openmm-protein-ligand-results:/results" \
  openmm-protein-ligand-runner:latest \
  -m prot_lig_runner.analysis --output /results --mmpbsa
```

## Diagnostics and tests

On failure, `failure_diagnostics.json` records the phase, last step and metrics, likely explanation, suggested next step, and any close atom pairs. During dynamics, the runner checks energy and kinetic temperature for non-finite values, extreme temperature, or large energy jumps. These checks identify warning signs; they do not prove a cause.

Run the unit tests in Docker:

```bash
docker run --rm -w /work -e PYTHONPATH=/work/src \
  -v "$PWD:/work:ro" --entrypoint python \
  openmm-protein-ligand-runner:latest \
  -m unittest discover -s tests -v
```
