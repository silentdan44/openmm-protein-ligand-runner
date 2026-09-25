FROM mambaorg/micromamba:2.0.5

WORKDIR /app
COPY --chown=$MAMBA_USER:$MAMBA_USER environment.yml /tmp/environment.yml
RUN micromamba create -y -f /tmp/environment.yml && \
    micromamba clean --all --yes

ENV PATH=/opt/conda/envs/openmm-protein-ligand-runner/bin:$PATH
COPY --chown=$MAMBA_USER:$MAMBA_USER run_prot_lig.py /app/run_prot_lig.py
COPY --chown=$MAMBA_USER:$MAMBA_USER input/ /app/input/

ENTRYPOINT ["python", "/app/run_prot_lig.py"]
CMD ["--help"]
