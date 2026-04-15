from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from kmwe.core.config_loader import ConfigError


def load_ingested_index(index_path: Path) -> dict[str, Any]:
    if not index_path.exists():
        raise ConfigError(f"ingested index 파일이 없습니다: {index_path}")
    data = json.loads(index_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ConfigError(f"ingested index는 dict여야 합니다: {index_path}")
    return data


def _resolve_shards_dir(index: dict[str, Any], index_path: Path) -> Path:
    raw = index.get("shards_dir")
    if raw:
        return Path(raw)
    return index_path.parent


def _normalize_shards(index: dict[str, Any]) -> dict[str, list[str]]:
    if isinstance(index.get("corpora"), dict):
        out: dict[str, list[str]] = {}
        for corpus, payload in index["corpora"].items():
            if isinstance(payload, dict) and isinstance(payload.get("shards"), list):
                out[str(corpus)] = [str(p) for p in payload["shards"]]
        if out:
            return out
    shards = index.get("shards")
    if isinstance(shards, dict):
        return {str(k): [str(p) for p in v] for k, v in shards.items()}
    if isinstance(shards, list):
        out: dict[str, list[str]] = {}
        for item in shards:
            if not isinstance(item, dict):
                continue
            corpus = str(item.get("corpus") or "")
            path = item.get("path") or item.get("shard")
            if corpus and path:
                out.setdefault(corpus, []).append(str(path))
        if out:
            return out
    raise ConfigError("ingested index에서 shards 정보를 찾지 못했습니다.")


def resolve_latest_ingest_artifact(
    cfg: dict[str, Any], *, stage_name: str = "ingest_corpus"
) -> tuple[Path, Path]:
    artifacts_dir = Path(cfg.get("paths", {}).get("artifacts_dir", "artifacts"))
    exp_id = str(cfg.get("exp", {}).get("exp_id", "default"))
    ingest_dir = artifacts_dir / exp_id / stage_name
    if not ingest_dir.exists():
        raise ConfigError(
            "ingest run 디렉터리를 찾지 못했습니다. "
            f"exp_id={exp_id} stage_name={stage_name} base_dir={ingest_dir}. "
            "index_path를 지정하세요."
        )
    run_dirs = sorted([p for p in ingest_dir.iterdir() if p.is_dir()], reverse=True)
    for run_dir in run_dirs:
        outputs_dir = run_dir / "outputs"
        if not outputs_dir.exists():
            continue
        for candidate in [
            outputs_dir / "ingested_index.json",
            outputs_dir / "ingest_index.json",
            outputs_dir / "ingest_corpus_index.json",
            outputs_dir / "index.json",
        ]:
            if candidate.exists():
                return candidate, outputs_dir
    raise ConfigError(
        "ingested index를 찾지 못했습니다. "
        f"exp_id={exp_id} stage_name={stage_name} base_dir={ingest_dir}. "
        "예시: --index_path artifacts/<exp_id>/{stage_name}/<run_id>/outputs/ingested_index.json"
    )


def list_shard_paths(index: dict[str, Any], *, index_path: Path) -> dict[str, list[Path]]:
    shards_by_corpus = _normalize_shards(index)
    shards_dir = _resolve_shards_dir(index, index_path)
    resolved: dict[str, list[Path]] = {}
    for corpus, paths in shards_by_corpus.items():
        resolved[corpus] = [
            Path(p) if Path(p).is_absolute() else (shards_dir / p) for p in paths
        ]
    return resolved
