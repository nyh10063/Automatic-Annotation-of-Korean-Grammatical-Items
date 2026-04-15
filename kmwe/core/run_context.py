from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any

from .utils import iso_now
from kmwe.utils.jsonio import write_json


class RunContext:
    def __init__(
        self,
        run_dir: Path,
        stage: str,
        run_id: str,
        exp_id: str,
        profile_id: str | None,
        project_root: str,
        config_sources: dict[str, Any],
        argv: list[str],
    ) -> None:
        self.run_dir = run_dir
        self.stage = stage
        self.run_id = run_id
        self.exp_id = exp_id
        self.profile_id = profile_id
        self.project_root = project_root
        self.config_sources = config_sources
        self.argv = argv
        self.logs_dir = self.run_dir / "logs"
        self.outputs_dir = self.run_dir / "outputs"

    def prepare_folders(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.outputs_dir.mkdir(parents=True, exist_ok=True)

    def manifest_path(self) -> Path:
        return self.run_dir / "manifest.json"

    def create_manifest(self) -> dict[str, Any]:
        manifest = {
            "stage": self.stage,
            "run_id": self.run_id,
            "created_at": iso_now(),
            "exp_id": self.exp_id,
            "profile_id": self.profile_id,
            "project_root": self.project_root,
            "config_sources": self.config_sources,
            "argv": self.argv,
            "git_commit": _get_git_commit(),
            "outputs_dir": str(self.outputs_dir),
            "logs_dir": str(self.logs_dir),
            "status": "running",
        }
        self._write_manifest(manifest)
        return manifest

    def update_manifest(self, updates: dict[str, Any]) -> dict[str, Any]:
        path = self.manifest_path()
        if path.exists():
            manifest = json.loads(path.read_text(encoding="utf-8"))
        else:
            manifest = {}
        manifest.update(updates)
        self._write_manifest(manifest)
        return manifest

    def setup_logging(self) -> logging.Logger:
        logger = logging.getLogger("kmwe")
        logger.setLevel(logging.INFO)
        logger.propagate = False
        if logger.handlers:
            return logger

        log_path = self.logs_dir / "stage.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8", delay=False)
        formatter = logging.Formatter(
            "%(asctime)s %(levelname)s %(message)s"
        )
        try:
            file_handler.stream.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
        except Exception:
            pass
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(formatter)
        logger.addHandler(console)

        return logger

    def write_error(self, exc: Exception) -> None:
        error_path = self.logs_dir / "errors.log"
        import traceback

        error_path.write_text(
            "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
            encoding="utf-8",
        )

    def _write_manifest(self, manifest: dict[str, Any]) -> None:
        write_json(self.manifest_path(), manifest, indent=2)


def _get_git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() or None
    except Exception:
        return None
