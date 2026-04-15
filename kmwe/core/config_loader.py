from __future__ import annotations

import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from .utils import ensure_absolute, find_upwards, is_colab_env, resolve_project_root_auto
from kmwe.utils.jsonio import write_json


ALLOWED_STAGES = {
    "validate_dict",
    "pos_mapping",
    "ingest_corpus",
    "ingest_train_corpora",
    "build_silver",
    "build_bgroup_sft",
    "train_llm_sft",
    "eval_hf_sft",
    "eval_openai_sft",
    "train_tapt",
    "train_mtl",
    "train_weak",
    "train_finetune",
    "train_bgroup_encoder_ce",
    "infer_step1",
    "infer_step2_rerank",
    "eval",
    "eval_b",
    "gate_check",
    "eval_rule_gold",
    "eval_rule_end_to_end",
}

PATH_KEYS_TO_ABSOLUTIZE = [
    "paths.project_root",
    "paths.data_dir",
    "paths.dict_xlsx",
    "paths.gold_xlsx",
    "paths.gold_b_xlsx",
    "paths.infer_input_csv",
    "paths.artifacts_dir",
    "paths.logs_dir",
    "paths.cache_dir",
    "paths.tmp_dir",
]

ENV_MAPPING = {
    "WANDB_PROJECT": "wandb.project",
    "WANDB_ENTITY": "wandb.entity",
    "WANDB_MODE": "wandb.mode",
    "WANDB_TAGS": "wandb.tags",
    "KMWE_DEVICE": "runtime.device",
    "KMWE_MP": "runtime.mixed_precision",
}


class ConfigError(RuntimeError):
    pass


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise ConfigError(f"설정 파일이 존재하지 않습니다: {path}") from exc
    try:
        data = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        raise ConfigError(f"YAML 파싱 오류: {path} - {exc}") from exc
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(f"설정 최상위는 dict여야 합니다: {path}")
    return data


def _get_by_path(cfg: dict[str, Any], path: str) -> Any:
    cur: Any = cfg
    for part in path.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


def _set_by_path(cfg: dict[str, Any], path: str, value: Any) -> None:
    cur: dict[str, Any] = cfg
    parts = path.split(".")
    for part in parts[:-1]:
        if part not in cur or not isinstance(cur[part], dict):
            cur[part] = {}
        cur = cur[part]
    cur[parts[-1]] = value


def _update_provenance(
    provenance: dict[str, dict[str, Any]],
    base_path: str,
    value: Any,
    source_label: str,
) -> None:
    if isinstance(value, dict):
        for key, sub_value in value.items():
            child_path = f"{base_path}.{key}" if base_path else key
            _update_provenance(provenance, child_path, sub_value, source_label)
        return
    provenance[base_path] = {"source": source_label, "value": value}


def _merge_dicts(
    base: dict[str, Any],
    override: dict[str, Any],
    provenance: dict[str, dict[str, Any]],
    source_label: str,
    base_path: str = "",
) -> None:
    for key, value in override.items():
        path = f"{base_path}.{key}" if base_path else key
        if key not in base:
            base[key] = deepcopy(value)
            _update_provenance(provenance, path, value, source_label)
            continue
        existing = base[key]
        if isinstance(existing, dict) and isinstance(value, dict):
            _merge_dicts(existing, value, provenance, source_label, path)
            continue
        if isinstance(existing, dict) != isinstance(value, dict):
            raise ConfigError(f"타입 충돌: {path} (dict vs scalar/list)")
        if isinstance(existing, list) != isinstance(value, list):
            raise ConfigError(f"타입 충돌: {path} (list vs scalar/dict)")
        base[key] = deepcopy(value)
        _update_provenance(provenance, path, value, source_label)


def _env_overrides() -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for env_key, cfg_path in ENV_MAPPING.items():
        if env_key not in os.environ:
            continue
        raw_value = os.environ[env_key]
        if env_key == "WANDB_TAGS":
            value: Any = [tag.strip() for tag in raw_value.split(",") if tag.strip()]
        else:
            value = raw_value
        _set_by_path(overrides, cfg_path, value)
    return overrides


def _resolve_local_config_path(config_dir: Path, cli_path: str | None) -> Path | None:
    if cli_path:
        path = Path(cli_path).expanduser()
        if not path.exists():
            raise ConfigError(f"local config 파일이 존재하지 않습니다: {path}")
        return path.resolve()

    local = config_dir / "local.yaml"
    local_colab = config_dir / "local_colab.yaml"
    if local.exists() and local_colab.exists() and not is_colab_env():
        raise ConfigError(
            "config/local.yaml과 config/local_colab.yaml이 모두 존재합니다. "
            "--config로 명시적으로 선택하세요."
        )
    if is_colab_env() and local_colab.exists():
        return local_colab.resolve()
    if local.exists():
        return local.resolve()
    return None


def _validate_required_files(cfg: dict[str, Any]) -> None:
    stage = str(_get_by_path(cfg, "run.stage") or "").strip()
    file_keys = ["paths.dict_xlsx", "paths.infer_input_csv"]

    if stage == "build_bgroup_sft":
        gold_key = "paths.gold_b_xlsx" if _get_by_path(cfg, "paths.gold_b_xlsx") not in (None, "") else "paths.gold_xlsx"
        file_keys.append(gold_key)
    elif stage in {"train_llm_sft", "eval_openai_sft"}:
        pass
    else:
        file_keys.append("paths.gold_xlsx")

    for key in file_keys:
        value = _get_by_path(cfg, key)
        if value in (None, ""):
            continue
        path = Path(str(value))
        if not path.exists():
            raise ConfigError(f"필수 파일이 존재하지 않습니다: {key}={path}")


def load_and_merge_config(
    exp_ids: list[str] | None,
    profile_id: str | None,
    local_config_path: str | None,
    cli_overrides: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, Any]]:
    default_path = find_upwards(Path.cwd(), "config/default.yaml")
    if default_path is None:
        raise ConfigError("config/default.yaml을 찾을 수 없습니다.")
    config_dir = default_path.parent
    sources: dict[str, Any] = {"default": str(default_path.resolve())}

    default_cfg = _read_yaml(default_path)
    provenance: dict[str, dict[str, Any]] = {}
    merged = deepcopy(default_cfg)
    _update_provenance(
        provenance, "", merged, f"default:{default_path.resolve()}"
    )

    exp_ids = exp_ids or []
    for exp_id in exp_ids:
        exp_path = (config_dir / "exp" / f"{exp_id}.yaml").resolve()
        if not exp_path.exists():
            raise ConfigError(f"exp 설정 파일이 존재하지 않습니다: {exp_path}")
        exp_cfg = _read_yaml(exp_path)
        _merge_dicts(
            merged, exp_cfg, provenance, f"exp:{exp_path}", ""
        )
        sources.setdefault("exp", []).append(str(exp_path))

    if profile_id:
        profile_path = (config_dir / "profiles" / f"{profile_id}.yaml").resolve()
        if not profile_path.exists():
            raise ConfigError(f"profile 설정 파일이 존재하지 않습니다: {profile_path}")
        profile_cfg = _read_yaml(profile_path)
        _merge_dicts(
            merged, profile_cfg, provenance, f"profile:{profile_path}", ""
        )
        sources["profile"] = str(profile_path)
    else:
        sources["profile"] = "missing"

    local_path = _resolve_local_config_path(config_dir, local_config_path)
    if local_path is not None:
        try:
            if local_path.resolve() == default_path.resolve():
                local_path = None
                sources["local"] = "ignored_same_as_default"
        except FileNotFoundError:
            pass
    if local_path:
        local_cfg = _read_yaml(local_path)
        _merge_dicts(
            merged, local_cfg, provenance, f"local:{local_path}", ""
        )
        sources["local"] = str(local_path)
    else:
        sources["local"] = "missing"

    env_cfg = _env_overrides()
    if env_cfg:
        _merge_dicts(
            merged, env_cfg, provenance, "env:ENV", ""
        )
        sources["env"] = "ENV"
    else:
        sources["env"] = "missing"

    if cli_overrides:
        _merge_dicts(
            merged, cli_overrides, provenance, "cli:CLI", ""
        )
        sources["cli"] = "CLI"
    else:
        sources["cli"] = "missing"

    exp_id_effective = exp_ids[-1] if exp_ids else "default"
    merged.setdefault("exp", {})
    if not isinstance(merged["exp"], dict):
        raise ConfigError("exp 설정은 dict여야 합니다.")
    merged["exp"]["exp_id"] = exp_id_effective
    if exp_ids:
        merged["exp"]["exp_ids"] = exp_ids
    _update_provenance(
        provenance, "exp.exp_id", exp_id_effective, "cli:exp_ids"
    )
    sources["exp_id_effective"] = exp_id_effective

    if "paths" not in merged or not isinstance(merged["paths"], dict):
        raise ConfigError("paths 설정이 누락되었습니다.")

    project_root_value = merged["paths"].get("project_root")
    if project_root_value == "__AUTO__":
        resolved_root = resolve_project_root_auto(default_path.parent)
        merged["paths"]["project_root"] = str(resolved_root.resolve())
    elif isinstance(project_root_value, str):
        merged["paths"]["project_root"] = ensure_absolute(
            project_root_value, default_path.parent
        )
    else:
        raise ConfigError(f"paths.project_root가 유효한 문자열이 아닙니다: {project_root_value}")

    project_root = Path(merged["paths"]["project_root"])
    for key in PATH_KEYS_TO_ABSOLUTIZE:
        value = _get_by_path(merged, key)
        if value in (None, ""):
            continue
        if not isinstance(value, str):
            raise ConfigError(f"경로 값이 문자열이 아닙니다: {key}={value}")
        if key == "paths.project_root":
            resolved = ensure_absolute(value, default_path.parent)
        else:
            resolved = ensure_absolute(value, project_root)
        _set_by_path(merged, key, resolved)
        if key in provenance:
            provenance[key]["value"] = resolved
        else:
            provenance[key] = {"source": "resolved", "value": resolved}

    artifacts_dir = _get_by_path(merged, "paths.artifacts_dir")
    if artifacts_dir:
        Path(str(artifacts_dir)).mkdir(parents=True, exist_ok=True)

    _validate_required_files(merged)

    return merged, provenance, sources


def write_config_outputs(
    run_dir: Path,
    config: dict[str, Any],
    provenance: dict[str, dict[str, Any]],
) -> None:
    resolved_path = run_dir / "config_resolved.yaml"
    prov_path = run_dir / "config_provenance.json"
    resolved_path.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    write_json(prov_path, provenance, indent=2)
