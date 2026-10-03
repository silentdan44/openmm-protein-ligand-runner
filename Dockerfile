FROM mambaorg/micromamba:2.0.5

USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake curl git libcurl4-openssl-dev libeigen3-dev \
    libfmt-dev libzip-dev nlohmann-json3-dev pkg-config unzip zlib1g-dev && \
    rm -rf /var/lib/apt/lists/*

# Build the standalone TmDet 4.0 implementation from its archived 4.1.2 release.
ARG TMDET_RELEASE=4.1.2
ARG TMDET_COMMIT=f8ceefe10a5eb78a3156869010b6107f934cc78b
ARG GEMMI_VERSION=0.7.0
ARG PUGIXML_VERSION=1.14
RUN curl -fsSL "https://github.com/project-gemmi/gemmi/archive/refs/tags/v${GEMMI_VERSION}.tar.gz" -o /tmp/gemmi.tar.gz && \
    tar -xzf /tmp/gemmi.tar.gz -C /tmp && \
    cmake -S "/tmp/gemmi-${GEMMI_VERSION}" -B "/tmp/gemmi-${GEMMI_VERSION}/build" \
      -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=OFF -DCMAKE_CXX_FLAGS=-fno-lto && \
    cmake --build "/tmp/gemmi-${GEMMI_VERSION}/build" -j2 && \
    cmake --install "/tmp/gemmi-${GEMMI_VERSION}/build" && \
    rm -rf /tmp/gemmi.tar.gz "/tmp/gemmi-${GEMMI_VERSION}"
RUN mkdir -p /tmp/contrib && \
    curl -fsSL "https://github.com/zeux/pugixml/archive/refs/tags/v${PUGIXML_VERSION}.tar.gz" -o /tmp/pugixml.tar.gz && \
    tar -xzf /tmp/pugixml.tar.gz -C /tmp/contrib && \
    ln -s "/tmp/contrib/pugixml-${PUGIXML_VERSION}" /tmp/contrib/pugixml && \
    mkdir -p /tmp/tmdet-compat && \
    printf '#pragma once\n#include <span>\n#define FMT_HEADER_ONLY\n#include <fmt/format.h>\nnamespace std { using ::fmt::format; template<class... Args> using format_string = ::fmt::format_string<Args...>; }\n' > /tmp/tmdet-compat/format && \
    git init /tmp/TmDet && \
    git -C /tmp/TmDet remote add origin https://github.com/brgenzim/TmDet.git && \
    git -C /tmp/TmDet fetch --depth 1 origin "${TMDET_COMMIT}" && \
    git -C /tmp/TmDet checkout FETCH_HEAD && \
    sed -i 's/ss << ymd;/ss << static_cast<int>(ymd.year()) << "-" << static_cast<unsigned>(ymd.month()) << "-" << static_cast<unsigned>(ymd.day());/' /tmp/TmDet/src/System/Date.hpp && \
    cmake -S /tmp/TmDet -B /tmp/TmDet/build -DCMAKE_CXX_FLAGS=-I/tmp/tmdet-compat && \
    cmake --build /tmp/TmDet/build -j2 && \
    cmake --install /tmp/TmDet/build && \
    rm -rf /tmp/pugixml.tar.gz /tmp/TmDet /tmp/contrib /tmp/tmdet-compat

RUN ldconfig

WORKDIR /app
COPY --chown=$MAMBA_USER:$MAMBA_USER environment.yml /tmp/environment.yml
RUN micromamba create -y -f /tmp/environment.yml && \
    micromamba clean --all --yes

# TmDet requires an environment file even when an input structure is supplied
# by path. Keep its downloaded CCD and cache data inside the container.
RUN mkdir -p /app /work/data/ccd /work/cache /home/data/wwPDB/status /home/data/wwPDB/data/structures/divided/pdb /home/data/wwPDB/data/structures/divided/updated_mmcif && \
    printf '%s\n' \
      'APP_NAME=tmdet' \
      'VERSION=4.1.1' \
      'PDB_DATA_ROOT=/home/data/wwPDB' \
      'PDB_STATUS_DIR=/home/data/wwPDB/status' \
      'PDB_DATA_DIR=/home/data/wwPDB/data' \
      'PDB_ENT_DIR=/home/data/wwPDB/data/structures/divided/pdb' \
      'PDB_CIF_DIR=/home/data/wwPDB/data/structures/divided/updated_mmcif' \
      'PDB_CIF_EXT=_updated.cif.gz' \
      'TMDET_DATA_ROOT=/work/data' \
      'TMDET_CACHE_ROOT=/work/cache' \
      'TMDET_CC_DIR=/work/data/ccd' \
      'TMDET_CC_FILE=/work/data/ccd/components.cif.gz' \
      'TMDET_CC_URL=https://files.wwpdb.org/pub/pdb/data/monomers/components.cif.gz' \
      'TMDET_POLYMER_FILTER_FILE=/work/data/polymer_filter.txt' \
      'TMDET_MIN_NUMBER_OF_RESIDUES_IN_CHAIN=15' \
      'TMDET_BALL_DIST=0.20' \
      'TMDET_SURF_PROBSIZE=1.4' \
      'TMDET_SURF_ZSLICE=0.05' \
      'TMDET_SURF_DIST=0.1' > /app/tmdet.env && \
    chown -R "$MAMBA_USER:$MAMBA_USER" /app/tmdet.env /work /home/data/wwPDB

ENV PATH=/opt/conda/envs/openmm-protein-ligand-runner/bin:$PATH
COPY --chown=$MAMBA_USER:$MAMBA_USER run_prot_lig.py /app/run_prot_lig.py
COPY --chown=$MAMBA_USER:$MAMBA_USER validate_trpv4_membrane.py /app/validate_trpv4_membrane.py
COPY --chown=$MAMBA_USER:$MAMBA_USER trajectory_analysis.py analyze_run.py /app/
COPY --chown=$MAMBA_USER:$MAMBA_USER input/ /app/input/

USER $MAMBA_USER

ENTRYPOINT ["python", "/app/run_prot_lig.py"]
CMD ["--help"]
