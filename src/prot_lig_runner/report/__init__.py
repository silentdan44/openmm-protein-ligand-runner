"""Modular run reports.

The report is assembled from a dataset plus a registry of figures, so the same
code produces a partial report from a running simulation and a final report
with publication figures.
"""

from .dataset import RunDataset
from .render import build_report, collect_figures

__all__ = ["RunDataset", "build_report", "collect_figures"]
