# OpenMM runner

Docker runner for protein–ligand simulations. `input/` has an AQP3 solution test pair and a TRPV4–capsaicin membrane test pair. The default run is the existing solution workflow; pass `--membrane` to build a membrane, water, and ions together with OpenMM's `Modeller.addMembrane()`.

## Build

```bash
docker build -t openmm-protein-ligand-runner .
```

The image pins the Python simulation dependencies and builds the standalone TmDet 4.0 method from release 4.1.2 for local receptor orientation.

## Solution run

```bash
mkdir -p "$HOME/openmm-protein-ligand-runner-results"
docker run --rm \
  -v "$HOME/openmm-protein-ligand-runner-results:/results" \
  openmm-protein-ligand-runner \
  --receptor /app/input/aqp3_solution_test_receptor.pdb \
  --ligand /app/input/aqp3_solution_test_ligand.sdf \
  --output /results
```

## Membrane run: TRPV4

The TmDet mode transforms both the receptor and ligand by the same rigid-body transform, preserving the binding pose:

```bash
docker run --rm \
  -v "$HOME/openmm-protein-ligand-runner-results:/results" \
  openmm-protein-ligand-runner \
  --receptor /app/input/trpv4_membrane_test_receptor.pdb \
  --ligand /app/input/trpv4_membrane_test_ligand.sdf \
  --output /results \
  --membrane --orientation tmdet --lipid-type POPC \
  --membrane-padding-nm 1.0 --ionic-strength-molar 0.15 \
  --nvt-steps 10 --npt-steps 10 --prod-steps 10 --minimize-iterations 10
```

Orientation choices:

- `--orientation tmdet` runs local TmDet and writes its oriented receptor (`tmdet_oriented.cif`) and annotation to the output directory.
- `--orientation preoriented` uses coordinates as supplied. The receptor must have the membrane plane in XY and its center at `--membrane-center-z-nm`; the ligand SDF must use the same coordinate frame.
- `--orientation manual` rotates the supplied membrane normal onto Z and translates the supplied point on the membrane center plane to `(0, 0, membrane-center-z)`. Example: `--orientation manual --manual-membrane-normal 0 0 1 --manual-membrane-center-angstrom 88 88 106`.

The first version supports one lipid type per bilayer (`POPC`, `POPE`, `DLPC`, `DLPE`, `DMPC`, `DOPC`, or `DPPC`). `Modeller.addMembrane()` creates both leaflets, water, and ions; do not add solvent separately. Membrane mode uses Amber ff14SB, Lipid17, TIP3P, and OpenFF ligand parameters, plus a membrane barostat during NPT.

The included TRPV4 ligand coordinates are already in the receptor frame. For other inputs, the SDF must describe the ligand pose in the supplied receptor coordinates; TmDet/manual orientation applies the same rigid transform to both.

## Inputs and run controls

Mount custom inputs read-only and pass their container paths with `--receptor` and `--ligand`. Simulation length, reporting intervals, pressure, temperature, timestep, seed, and platform are configurable through the corresponding CLI flags. For a long run, mount an external results directory and use `--resume` with identical inputs and settings; the runner fingerprints membrane and orientation settings too.

The short TRPV4 command above checks software execution and basic membrane construction, not scientific convergence or force-field validation. Inspect `system_final.pdb`, `system.gro`, and the trajectory to confirm the protein and ligand are embedded as intended before production runs.

## Diagnostics and MMPBSA

After production, the runner writes `analysis/diagnostics.png`, `analysis/protein_rmsf.png`, `analysis/metrics.json`, and `analysis/report.html`. The plots show temperature and energy across NVT/NPT/production, NPT density and box volume, protein-backbone and ligand RMSD, protein/ligand radius of gyration, protein–ligand residue contacts, and protein Cα RMSF. Coordinates are made whole and the protein is fitted before structural metrics are calculated; RMSD/RMSF/Rg/contact plots and the prepared MMPBSA trajectory use production frames only. The analysis loads every 10th saved trajectory frame by default to limit memory; use `--analysis-stride 1` to analyze every saved frame. Temperature and NPT density/volume are diagnostics, not proof of convergence. OpenMM's `StateDataReporter` does not report instantaneous pressure, so the barostat plot checks volume and density against the configured pressure target.

Rebuild the report later, or start the optional end-state calculation, from an existing result directory:

```bash
docker run --rm --entrypoint python \
  -v "$HOME/openmm-protein-ligand-runner-results:/results" \
  openmm-protein-ligand-runner /app/analyze_run.py --output /results

docker run --rm --entrypoint python \
  -v "$HOME/openmm-protein-ligand-runner-results:/results" \
  openmm-protein-ligand-runner /app/analyze_run.py --output /results --mmpbsa \
  --mmpbsa-model gb --mmpbsa-interval 2
```

Alternatively, pass `--mmpbsa` to the simulation command to run the default single-trajectory GB calculation immediately after MD. Use `--mmpbsa-model pb` for PB. GB/PB without a membrane model does not represent the bilayer dielectric; for a membrane PB slab, choose `--mmpbsa-model pb_mem` and provide `--mmpbsa-membrane-thickness-angstrom`; `--mmpbsa-membrane-dielectric` defaults to 7.0. `gmx_MMPBSA` needs GROMACS and AmberTools, included in the expanded Conda image, and uses the OpenMM-exported complex topology plus `analysis/index.ndx` and the fitted trajectory. Results and logs are in `analysis/mmpbsa/`.

MMPBSA is an approximate end-state estimate, not an absolute binding free energy or a substitute for convergence checks. The OpenMM-to-GROMACS topology conversion, especially for OpenFF ligands and membrane force fields, must be validated for each system; inspect `gmx_MMPBSA.log` and confirm the generated topology/index selections before interpreting results. Increase sampling or run independent replicas for quantitative comparisons.

The end-to-end Docker smoke test completed on the included AQP3 ligand example with `--nvt-steps 20 --npt-steps 20 --prod-steps 200 --mmpbsa --mmpbsa-model gb --mmpbsa-interval 1`. OpenMM completed and `gmx_MMPBSA` 1.7.0 exited successfully after processing 11 frames; `FINAL_RESULTS_MMPBSA.dat` and `.csv` were written. This short run checks execution only, not equilibration or convergence. The GROMACS topology is exported without bond constraints so ParmEd retains constrained H–X bond connectivity for Amber topology and PB radii conversion; OpenMM dynamics still use H-bond constraints.

Run the unit tests with:

```bash
docker run --rm -w /work \
  -v "$PWD:/work:ro" \
  --entrypoint python openmm-protein-ligand-runner \
  -m unittest discover -s tests -v
```
