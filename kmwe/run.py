from __future__ import annotations

import argparse
import inspect
import sys
from pathlib import Path
from typing import Any

import yaml

from .core.config_loader import ALLOWED_STAGES, ConfigError, load_and_merge_config, write_config_outputs
from .core.run_context import RunContext
from .core.stage_registry import get_stage
import kmwe
import kmwe.stages  # noqa: F401
from .core.utils import flatten_list, generate_run_id


def _parse_set_overrides(items: list[str] | None) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    if not items:
        return overrides
    for item in items:
        if "=" not in item:
            raise ConfigError(f"--set 형식이 올바르지 않습니다: {item}")
        key, raw_value = item.split("=", 1)
        if not key:
            raise ConfigError(f"--set 키가 비어있습니다: {item}")
        value = yaml.safe_load(raw_value)
        _apply_dot_path(overrides, key, value)
    return overrides


def _apply_dot_path(target: dict[str, Any], path: str, value: Any) -> None:
    cur = target
    parts = path.split(".")
    for part in parts[:-1]:
        if part not in cur or not isinstance(cur[part], dict):
            cur[part] = {}
        cur = cur[part]
    cur[parts[-1]] = value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="KMWE stage runner")
    parser.add_argument("--stage", required=True, help="실행할 stage")
    parser.add_argument("--profile", help="profile id")
    parser.add_argument("--exp", action="append", nargs="+", help="exp id")
    parser.add_argument("--exp_id", help="exp id (alias of --exp)")
    parser.add_argument("--config", help="local override config path")
    parser.add_argument("--run_id", help="run id (미지정 시 자동 생성)")
    parser.add_argument("--seed", type=int, help="random seed override")
    parser.add_argument("--set", action="append", help="dot path override (a.b.c=value)")
    parser.add_argument("--source_name", help="ingest_corpus 전용 source_name override")
    parser.add_argument("--artifacts_dir", help="alias of --set paths.artifacts_dir=...")
    return parser


def _log_runtime_doctor(logger, stage_name: str) -> None:
    # keep this function extremely defensive: never crash
    try:
        import kmwe as _kmwe
        logger.info("doctor: kmwe.__file__=%s", getattr(_kmwe, "__file__", "N/A"))
    except Exception as e:
        logger.warning("doctor: failed to import kmwe: %r", e)

    try:
        from kmwe.core import stage_registry as _sr
        logger.info(
            "doctor: stage_registry.__file__=%s", getattr(_sr, "__file__", "N/A")
        )
        try:
            stages = _sr.list_stages()
            logger.info("doctor: registered_stages(n=%d)=%s", len(stages), stages)
        except Exception as e:
            logger.warning("doctor: list_stages failed: %r", e)

        try:
            fn = _sr.get_stage(stage_name)
            logger.info("doctor: stage=%s callable=%r", stage_name, fn)
            try:
                src = inspect.getsourcefile(fn) or "N/A"
                logger.info("doctor: stage=%s callable_source=%s", stage_name, src)
            except Exception as e:
                logger.warning("doctor: inspect callable_source failed: %r", e)
            try:
                logger.info(
                    "doctor: stage=%s callable_module=%s",
                    stage_name,
                    getattr(fn, "__module__", "N/A"),
                )
            except Exception:
                pass
        except Exception as e:
            logger.warning("doctor: get_stage(%s) failed: %r", stage_name, e)

    except Exception as e:
        logger.warning("doctor: failed to import stage_registry: %r", e)


def main(argv: list[str] | None = None) -> int:
    raw_argv = sys.argv[1:]
    config_flag_count = sum(
        1
        for token in raw_argv
        if token == "--config" or token.startswith("--config=")
    )
    has_multiple_config_flags = config_flag_count >= 2

    parser = build_parser()
    args = parser.parse_args(argv)
    alias_used: list[str] = []

    if args.stage not in ALLOWED_STAGES:
        allowed = ", ".join(sorted(ALLOWED_STAGES))
        sys.stderr.write(
            f"허용되지 않은 stage입니다: {args.stage}\n허용 stage 목록: {allowed}\n"
        )
        return 2

    if args.exp_id:
        args.exp = (args.exp or []) + [[args.exp_id]]
        alias_used.append("--exp_id -> --exp")
    exp_ids = flatten_list(args.exp)
    cli_overrides: dict[str, Any] = {}
    if exp_ids:
        cli_overrides["exp"] = {"exp_id": exp_ids[-1]}
    cli_overrides["run"] = {"stage": args.stage}
    if args.seed is not None:
        cli_overrides["runtime"] = {"seed": args.seed}
    set_overrides = _parse_set_overrides(args.set)
    if set_overrides:
        cli_overrides = _merge_cli(cli_overrides, set_overrides)
    if args.artifacts_dir:
        _apply_dot_path(cli_overrides, "paths.artifacts_dir", args.artifacts_dir)
        alias_used.append("--artifacts_dir -> paths.artifacts_dir")

    local_config_path = args.config
    if local_config_path:
        if Path(local_config_path).name == "default.yaml":
            sys.stderr.write(
                "경고: --config default.yaml은 로컬 오버라이드가 아니므로 무시합니다.\n"
            )
            local_config_path = None
    if local_config_path is None:
        if args.profile and str(args.profile).startswith("colab_"):
            if Path("config/local_colab.yaml").exists():
                local_config_path = "config/local_colab.yaml"
            elif Path("config/local_colab.example.yaml").exists():
                local_config_path = "config/local_colab.example.yaml"
        if args.profile and str(args.profile).startswith("local_") and local_config_path is None:
            if Path("config/local.yaml").exists():
                local_config_path = "config/local.yaml"

    override_logs: list[str] = []
    if args.set:
        override_logs.extend(args.set)
    if args.artifacts_dir:
        override_logs.append(f"paths.artifacts_dir={args.artifacts_dir}")
    if args.seed is not None:
        override_logs.append(f"runtime.seed={args.seed}")
    if args.source_name is not None and args.stage == "ingest_corpus":
        override_logs.append(f"ingest.source_name={args.source_name}")
    try:
        cfg, provenance, sources = load_and_merge_config(
            exp_ids=exp_ids,
            profile_id=args.profile,
            local_config_path=local_config_path,
            cli_overrides=cli_overrides,
        )
    except ConfigError as exc:
        sys.stderr.write(f"설정 로드 실패: {exc}\n")
        return 1
    if "runtime" in cli_overrides and isinstance(cli_overrides.get("runtime"), dict):
        if "device" in cli_overrides["runtime"]:
            cfg.setdefault("runtime", {})["device"] = cli_overrides["runtime"]["device"]

    if args.stage == "ingest_corpus" and args.source_name is not None:
        source_name_value = str(args.source_name).strip()
        if not source_name_value:
            source_name_value = "unknown"
            sys.stderr.write("경고: --source_name이 비어 있어 'unknown'으로 보정합니다.\n")
        cfg.setdefault("ingest", {})["source_name"] = source_name_value

    run_id = args.run_id or generate_run_id()
    exp_id = cfg.get("exp", {}).get("exp_id", "default")
    artifacts_dir = Path(cfg["paths"]["artifacts_dir"])
    base_run_id = run_id
    run_dir = artifacts_dir / exp_id / args.stage / run_id
    dup_idx = 0
    while run_dir.exists():
        dup_idx += 1
        run_id = f"{base_run_id}_dup{dup_idx}"
        run_dir = artifacts_dir / exp_id / args.stage / run_id

    context = RunContext(
        run_dir=run_dir,
        stage=args.stage,
        run_id=run_id,
        exp_id=exp_id,
        profile_id=args.profile,
        project_root=cfg["paths"]["project_root"],
        config_sources=sources,
        argv=[sys.executable, "-m", "kmwe.run", *sys.argv[1:]],
    )
    context.prepare_folders()
    write_config_outputs(run_dir, cfg, provenance)
    context.create_manifest()

    logger = context.setup_logging()
    logger.info("[runner][config] base=default.yaml")
    logger.info("[runner][config] exps=%s", exp_ids)
    logger.info("[runner][config] overrides=%s", override_logs)
    logger.info(
        "[runner][resolved] stage=%s run_id=%s run_dir=%s",
        args.stage,
        run_id,
        run_dir,
    )
    logger.info(
        "[runner][resolved] artifacts_dir=%s outputs_dir=%s exp_id=%s",
        artifacts_dir,
        context.outputs_dir,
        exp_id,
    )
    logger.info(
        "[runner][resolved] artifacts_dir=%s exp_id=%s outputs_dir=%s",
        artifacts_dir,
        exp_id,
        context.outputs_dir,
    )
    if has_multiple_config_flags:
        logger.warning(
            "[cli][warning] multiple --config detected; argparse keeps only the last one"
        )
        logger.warning(
            "[cli][hint] use --profile colab_* + --exp ... (or single --config)"
        )
    if alias_used:
        logger.warning("[cli] alias used: %s", ", ".join(alias_used))
    logger.info("[run][ssot] kmwe.__file__=%s", getattr(kmwe, "__file__", "N/A"))
    logger.info("[run][ssot] run.py=%s", Path(__file__).resolve())
    logger.info(
        "[run][ssot] stage=%s exp=%s artifacts_dir=%s config_files=%s set=%s",
        args.stage,
        exp_id,
        artifacts_dir,
        sources,
        args.set or [],
    )
    logger.info(
        "stage=%s run_id=%s exp_id=%s artifacts_dir=%s",
        args.stage,
        run_id,
        exp_id,
        artifacts_dir,
    )

    try:
        stage_fn = get_stage(args.stage)
    except Exception as exc:
        context.write_error(exc)
        context.update_manifest(
            {
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        logger.error("stage 로드 실패(get_stage): %s", exc)
        _log_runtime_doctor(logger, args.stage)
        return 1

    _log_runtime_doctor(logger, args.stage)
    try:
        stage_fn(cfg=cfg, run_context=context)
    except Exception as exc:
        context.write_error(exc)
        context.update_manifest(
            {
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        logger.error("stage 실패: %s: %s", type(exc).__name__, exc)
        _log_runtime_doctor(logger, args.stage)
        return 1

    context.update_manifest({"status": "ok"})
    logger.info("stage 완료")
    return 0


def _merge_cli(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    merged = base.copy()
    for key, value in extra.items():
        if key not in merged:
            merged[key] = value
            continue
        if isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _merge_cli(merged[key], value)
            continue
        merged[key] = value
    return merged


if __name__ == "__main__":
    sys.exit(main())
