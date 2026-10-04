# OpenMM runner

Docker runner for protein–ligand simulations. `tests/fixtures/` has an AQP3 solution test pair and a TRPV4–capsaicin membrane test pair. The default run is the existing solution workflow; pass `--membrane` to build a membrane, water, and ions together with OpenMM's `Modeller.addMembrane()`.

## Project layout

```text
src/prot_lig_runner/   Simulation, diagnostics, and analysis package
tests/                 Unit tests
tests/fixtures/        Test receptor and ligand structures
tests/                 Unit tests
Dockerfile             Reproducible runtime image
environment.yml        Conda runtime dependencies
environment.cuda.yml   CUDA runtime dependencies for NVIDIA GPU servers
docker-compose.yml     Local run configuration
```

## Build

```bash
docker build -t openmm-protein-ligand-runner .
```

The image pins the Python simulation dependencies and builds the standalone TmDet 4.0 method from release 4.1.2 for local receptor orientation.

## NVIDIA GPU image

Build the optional CUDA 12 image on an NVIDIA Linux server (or build it for that server's architecture):

```bash
docker build --build-arg OPENMM_ENV_FILE=environment.cuda.yml \
  -t openmm-protein-ligand-runner:cuda .
```

The GPU image uses the same OpenMM version as the CPU image and adds the CUDA runtime. The host needs a supported NVIDIA driver and NVIDIA Container Toolkit configured for Docker. Pass the GPU through and select CUDA explicitly:

```bash
docker run --rm --gpus all \
  -v "$HOME/openmm-protein-ligand-runner-results:/results" \
  openmm-protein-ligand-runner:cuda \
  --receptor /app/tests/fixtures/aqp3_solution_test_receptor.pdb \
  --ligand /app/tests/fixtures/aqp3_solution_test_ligand.sdf \
  --output /results --platform CUDA
```

Verify that the container can access CUDA before a long run:

```bash
docker run --rm --gpus all --entrypoint python \
  openmm-protein-ligand-runner:cuda -m openmm.testInstallation
```

The Compose GPU service is opt-in so ordinary `docker compose up` continues to use the CPU image:

```bash
docker compose --profile gpu build openmm-protein-ligand-runner-cuda
docker compose --profile gpu run --rm openmm-protein-ligand-runner-cuda
```

## Solution run

```bash
mkdir -p "$HOME/openmm-protein-ligand-runner-results"
docker run --rm \
  -v "$HOME/openmm-protein-ligand-runner-results:/results" \
  openmm-protein-ligand-runner \
  --receptor /app/tests/fixtures/aqp3_solution_test_receptor.pdb \
  --ligand /app/tests/fixtures/aqp3_solution_test_ligand.sdf \
  --output /results
```

## Membrane run: TRPV4

The TmDet mode transforms both the receptor and ligand by the same rigid-body transform, preserving the binding pose:

```bash
docker run --rm \
  -v "$HOME/openmm-protein-ligand-runner-results:/results" \
  openmm-protein-ligand-runner \
  --receptor /app/tests/fixtures/trpv4_membrane_test_receptor.pdb \
  --ligand /app/tests/fixtures/trpv4_membrane_test_ligand.sdf \
  --output /results \
  --membrane --orientation tmdet --lipid-type POPC \
  --membrane-padding-nm 1.0 --ionic-strength-molar 0.15 \
  --nvt-steps 10 --npt-steps 10 --prod-steps 10 --minimize-iterations 10 \
  --report-interval 1 --xtc-interval 1 --checkpoint-interval 10 --skip-gromacs-export
```

Orientation choices:

- `--orientation tmdet` runs local TmDet and writes its oriented receptor (`tmdet_oriented.cif`) and annotation to the output directory.
- `--orientation preoriented` uses coordinates as supplied. The receptor must have the membrane plane in XY and its center at `--membrane-center-z-nm`; the ligand SDF must use the same coordinate frame.
- `--orientation manual` rotates the supplied membrane normal onto Z and translates the supplied point on the membrane center plane to `(0, 0, membrane-center-z)`. Example: `--orientation manual --manual-membrane-normal 0 0 1 --manual-membrane-center-angstrom 88 88 106`.

The first version supports one lipid type per bilayer (`POPC`, `POPE`, `DLPC`, `DLPE`, `DMPC`, `DOPC`, or `DPPC`). `Modeller.addMembrane()` creates both leaflets, water, and ions; do not add solvent separately. Membrane mode uses Amber ff14SB, Lipid17, TIP3P, and OpenFF ligand parameters, plus a membrane barostat during NPT.

The included TRPV4 ligand coordinates are already in the receptor frame. For other inputs, the SDF must describe the ligand pose in the supplied receptor coordinates; TmDet/manual orientation applies the same rigid transform to both.

## Inputs and run controls

Mount custom inputs read-only and pass their container paths with `--receptor` and `--ligand`. Simulation length, reporting intervals, pressure, temperature, timestep, seed, and platform are configurable through the corresponding CLI flags. `--platform auto` probes usable OpenMM contexts in CUDA → OpenCL → CPU order and falls back when a platform cannot initialize; a named platform is checked directly and reports an error if unavailable. The selected platform is used for MD; OpenMM's `Modeller.addMembrane()` chooses its own platform for its internal minimization. For a long run, mount an external results directory and use `--resume` with identical inputs and settings; the runner fingerprints membrane and orientation settings too.

The short TRPV4 command above checks software execution and basic membrane construction, not scientific convergence or force-field validation. It skips the optional, expensive GROMACS conversion; omit `--skip-gromacs-export` when `system.top` and `system.gro` are needed. Inspect `system_final.pdb`, `membrane_geometry.json`, and the trajectory to confirm the protein and ligand are embedded as intended before production runs.

To check that a short TRPV4 run completed dynamics and produced a bilayer, water, both lipid leaflets, and an aligned protein–ligand pose, use the repository's validator from Docker after the runner exits successfully:

```bash
docker run --rm --entrypoint python \
  -v "$HOME/openmm-protein-ligand-runner-results:/results:ro" \
  openmm-protein-ligand-runner \
  -m prot_lig_runner.validate_trpv4_membrane /results
```

The validator requires `run_state.json` to report a completed run and checks `membrane_geometry.json`, the final PDB, and non-empty trajectory. It is a structural smoke check, not a substitute for visual inspection or scientific validation.

The runner validates the input files, requires one ligand with a 3D conformer, and checks all step counts and intervals before simulation. It periodically writes a checkpoint for the active phase (`checkpoint-nvt.chk`, `checkpoint-npt.chk`, or `checkpoint-prod.chk`) and stores the active phase and input/configuration fingerprint in `run_state.json`. To continue an interrupted run, use the same input files and simulation settings, keep the same results directory, and add `--resume`. Checkpoints require the same OpenMM version and platform. The trajectory and CSV continue in append mode after resume; a completed run writes `final_state.xml`.

For Docker Compose, set `RESULTS_DIR` to a host directory outside this project. Its default is `../openmm-protein-ligand-runner-results`.

## Stability diagnostics

The runner warns about possible severe nonbonded atom overlaps in prepared coordinates. During dynamics it checks reported energy and kinetic temperature for non-finite values, extreme temperature, or a large energy jump. If a run fails, `failure_diagnostics.json` records the phase, last step and metrics, likely explanation, suggested next step, original error, and any close atom pairs. These checks identify warning signs; they do not prove the cause. Overlap distances use a conservative covalent-radius threshold, and close pairs should be inspected before changing the structure.

Run the dependency-light diagnostic checks in the project Docker image:

```bash
docker run --rm -w /work -e PYTHONPATH=/work/src -v "$PWD:/work:ro" \
  --entrypoint python openmm-protein-ligand-runner \
  -m unittest discover -s tests -v
```

## Trajectory analysis and MMPBSA

After a completed run, analysis writes `analysis/metrics.json`, `analysis/time_series.csv`, available tables of ligand contacts and hydrogen bonds, and `analysis/report.html`. The report contains simulation QC (temperature, potential energy, NPT density and box volume), protein stability (whole-protein and membrane-region RMSD when available, plus Cα RMSF), ligand pose and pocket contacts, contact/hydrogen-bond occupancy, and membrane area per lipid, phosphate-plane thickness, and leaflet counts when a membrane is present. Missing optional data suppresses only its plot and adds a note to the report.

The membrane-region Cα selection is derived from the mean phosphate z-planes in `system_final.pdb`; the report records the planes and selection count. This selects residues geometrically within the bilayer, without hard-coded channel residue numbers. For systems without a membrane, the membrane-specific RMSD and bilayer plots are omitted.

To plot distances between selected residues, repeat `--distance-pair CHAIN:RESID CHAIN:RESID`. Each plotted value is the Cα–Cα distance in nm. Chain IDs and residue numbers refer to `system_final.pdb`; use `_` for a blank chain ID. The flag can be passed to the simulation command or post-run analyzer:

```bash
docker run --rm --entrypoint python \
  -v "$HOME/openmm-protein-ligand-runner-results:/results" \
  openmm-protein-ligand-runner -m prot_lig_runner.analyze_run --output /results \
  --distance-pair A:550 A:600 --distance-pair A:550 B:600
```

Structural metrics and the prepared MMPBSA trajectory use production frames; every 10th saved frame is loaded by default to limit memory (`--analysis-stride 1` uses all frames). Density and volume are equilibration diagnostics, not proof of convergence.

Rebuild the report or run the optional end-state calculation from an existing result:

```bash
docker run --rm --entrypoint python \
  -v "$HOME/openmm-protein-ligand-runner-results:/results" \
  openmm-protein-ligand-runner -m prot_lig_runner.analyze_run --output /results

docker run --rm --entrypoint python \
  -v "$HOME/openmm-protein-ligand-runner-results:/results" \
  openmm-protein-ligand-runner -m prot_lig_runner.analyze_run --output /results --mmpbsa \
  --mmpbsa-model gb --mmpbsa-interval 2
```

Pass `--mmpbsa` to the simulation command to run single-trajectory MMPBSA after MD. Choose `--mmpbsa-model gb` or `pb`; for a membrane PB slab use `pb_mem` and provide `--mmpbsa-membrane-thickness-angstrom`. The membrane dielectric defaults to 7.0. MMPBSA gives an approximate end-state estimate, not an absolute binding free energy.

The MD trajectory contains the full protein, ligand, membrane, water, and ions. MMPBSA uses a separate protein-ligand-only topology and streams those atoms from the fitted trajectory; water and lipids are represented by the selected implicit GB/PB model. This keeps the MMPBSA topology small while retaining the full membrane environment during dynamics. A membrane run writes `mmpbsa_complex.pdb`, `.top`, and `.gro`; `analysis/mmpbsa/complex.xtc` contains the extracted production frames used by gmx_MMPBSA.

When MMPBSA frame results are available, the report plots energy terms by analyzed frame and their mean ± frame standard deviation. Add `--mmpbsa-decomposition` with `--mmpbsa` to calculate and plot the largest per-residue contributions; decomposition takes additional computation. Frame variation is descriptive because neighboring frames are correlated, and does not replace independent replicas.

GROMACS and AmberTools are included for this optional analysis. `--mmpbsa` cannot be combined with `--skip-gromacs-export`. For older results that lack `mmpbsa_complex.*`, post-run analysis can use the full GROMACS topology and trajectory when those files are present.
