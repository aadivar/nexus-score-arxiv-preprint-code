"""Canonical filesystem layout. One source of truth for every artifact path."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Layout:
    root: Path = PROJECT_ROOT

    @property
    def config_dir(self) -> Path:
        return self.root / "config"

    @property
    def data_dir(self) -> Path:
        return self.root / "data"

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def snapshot_dir(self) -> Path:
        return self.data_dir / "snapshots"

    @property
    def corpus_dir(self) -> Path:
        return self.data_dir / "corpus"

    @property
    def runs_dir(self) -> Path:
        return self.data_dir / "runs"

    @property
    def derived_dir(self) -> Path:
        return self.data_dir / "derived"

    # ---- config files ----
    @property
    def preregistration_yaml(self) -> Path:
        return self.config_dir / "preregistration.yaml"

    @property
    def nexus_weights_yaml(self) -> Path:
        return self.config_dir / "nexus_weights.yaml"

    @property
    def domains_yaml(self) -> Path:
        return self.config_dir / "domains.yaml"

    # ---- snapshot-versioned paths ----
    def openalex_cache(self, snapshot_label: str) -> Path:
        return self.snapshot_dir / "openalex" / snapshot_label

    def corpus_version(self, version: str) -> Path:
        return self.corpus_dir / version


LAYOUT = Layout()
