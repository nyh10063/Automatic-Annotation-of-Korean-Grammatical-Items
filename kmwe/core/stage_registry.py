from __future__ import annotations

from typing import Callable, Dict

from .config_loader import ALLOWED_STAGES


StageCallable = Callable[..., object]

_STAGE_REGISTRY: Dict[str, StageCallable] = {}


def register_stage(name: str) -> Callable[[StageCallable], StageCallable]:
    if name not in ALLOWED_STAGES:
        raise ValueError(f"허용되지 않은 stage 등록 시도: {name}")

    def decorator(func: StageCallable) -> StageCallable:
        _STAGE_REGISTRY[name] = func
        return func

    return decorator



def get_stage(name: str) -> StageCallable:
    if name not in _STAGE_REGISTRY:
        registered = ", ".join(sorted(_STAGE_REGISTRY.keys()))
        raise KeyError(
            f"등록되지 않은 stage: {name}\n"
            f"등록된 stage 목록: {registered}\n"
            "힌트: 공개용 repo에서는 A/B 추론·평가에 필요한 stage만 등록되어 있습니다."
        )
    return _STAGE_REGISTRY[name]



def list_stages() -> list[str]:
    return sorted(_STAGE_REGISTRY.keys())


@register_stage("validate_dict")
def stage_validate_dict(*, cfg, run_context, **_kwargs):
    from kmwe.stages.validate_dict import run_validate_dict

    return run_validate_dict(cfg=cfg, run_context=run_context)


@register_stage("train_bgroup_encoder_ce")
def stage_train_bgroup_encoder_ce(*, cfg, run_context, **_kwargs):
    from kmwe.stages.train_bgroup_encoder_ce import run_train_bgroup_encoder_ce

    return run_train_bgroup_encoder_ce(cfg=cfg, run_context=run_context)


@register_stage("infer_step1")
def stage_infer_step1(*, cfg, run_context, **_kwargs):
    from kmwe.stages.infer_step1 import run_infer_step1

    return run_infer_step1(cfg=cfg, run_context=run_context)


@register_stage("eval")
def stage_eval(*, cfg, run_context, **_kwargs):
    from kmwe.stages.eval import run_eval

    return run_eval(cfg=cfg, run_context=run_context)


@register_stage("eval_rule_gold")
def stage_eval_rule_gold(*, cfg, run_context, **_kwargs):
    from kmwe.stages.eval_rule_gold import run_eval_rule_gold

    return run_eval_rule_gold(cfg=cfg, run_context=run_context)


@register_stage("eval_rule_end_to_end")
def stage_eval_rule_end_to_end(*, cfg, run_context, **_kwargs):
    from kmwe.stages.eval_rule_end_to_end import run_eval_rule_end_to_end

    return run_eval_rule_end_to_end(cfg=cfg, run_context=run_context)
