# OpenMM runner

Docker runner for protein–ligand simulations with OpenMM. `input/AQP3.pdb` and `input/rank1.sdf` are included as an example system. The Python environment, including `openmmforcefields`, is installed in the container from `environment.yml`.

## Build

```bash
docker build -t openmm-protein-ligand-runner .
```

## Run the example

Choose a host directory for the results; it is mounted outside the project:

```bash
mkdir -p "$HOME/openmm-protein-ligand-runner-results"
docker run --rm \
  -v "$HOME/openmm-protein-ligand-runner-results:/results" \
  openmm-protein-ligand-runner \
  --receptor /app/input/AQP3.pdb \
  --ligand /app/input/rank1.sdf \
  --output /results
```

The default production stage is 500 ns. For a short smoke run, reduce the step counts and minimization iterations:

```bash
docker run --rm \
  -v "$HOME/openmm-protein-ligand-runner-results:/results" \
  openmm-protein-ligand-runner \
  --receptor /app/input/AQP3.pdb \
  --ligand /app/input/rank1.sdf \
  --output /results \
  --nvt-steps 10 --npt-steps 10 --prod-steps 10 --minimize-iterations 10
```

## Run another system

Mount the input directory read-only and pass the paths inside the container:

```bash
docker run --rm \
  -v "$PWD/my-system:/data:ro" \
  -v "$HOME/openmm-protein-ligand-runner-results:/results" \
  openmm-protein-ligand-runner \
  --receptor /data/receptor.pdb \
  --ligand /data/ligand.sdf \
  --output /results
```

The runner validates the input files, requires one ligand with a 3D conformer, and checks all step counts and intervals before simulation. It accepts `--receptor`, `--ligand`, `--output`, `--nvt-steps`, `--npt-steps`, `--prod-steps`, `--report-interval`, `--xtc-interval`, `--checkpoint-interval`, and `--minimize-iterations`.

Temperature, pressure, friction, timestep, solvent padding, ionic strength, random seed, and platform are configurable with `--temperature-k`, `--pressure-bar`, `--friction-per-ps`, `--timestep-fs`, `--padding-nm`, `--ionic-strength-molar`, `--seed`, and `--platform`. The `auto` platform choice checks CUDA, OpenCL, CPU, then Reference. The included conda-forge image is intended for CPU; GPU use needs a compatible GPU-enabled runtime and OpenMM build.

The runner periodically writes a checkpoint for the active phase (`checkpoint-nvt.chk`, `checkpoint-npt.chk`, or `checkpoint-prod.chk`) and stores the active phase and input/configuration fingerprint in `run_state.json`. To continue an interrupted run, use the same input files and simulation settings, keep the same results directory, and add `--resume`:

```bash
docker run --rm \
  -v "$HOME/openmm-protein-ligand-runner-results:/results" \
  openmm-protein-ligand-runner \
  --receptor /app/input/AQP3.pdb \
  --ligand /app/input/rank1.sdf \
  --output /results \
  --resume
```

Checkpoints require the same OpenMM version and platform. The runner checks the recorded settings and input file hashes before loading one. The trajectory and CSV continue in append mode after resume. A completed run writes `final_state.xml`; `--resume` is only for incomplete runs.

For Docker Compose, set `RESULTS_DIR` to a host directory outside this project. Its default is `../openmm-protein-ligand-runner-results`.

## Stability diagnostics

The runner warns about possible severe nonbonded atom overlaps in prepared coordinates. During dynamics it checks reported energy and kinetic temperature for non-finite values, extreme temperature, or a large energy jump. If a run fails, `failure_diagnostics.json` records the phase, last step and metrics, likely explanation, suggested next step, original error, and any close atom pairs. These checks identify warning signs; they do not prove the cause. Overlap distances use a conservative covalent-radius threshold, and close pairs should be inspected before changing the structure.

Run the dependency-light diagnostic checks with:

```bash
python -m unittest discover -s tests -v
```
