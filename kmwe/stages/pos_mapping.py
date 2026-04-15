from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any

from kmwe.core.run_context import RunContext
from kmwe.utils.jsonio import write_json

# validate_dict stage callable signature를 그대로 따름

DIRECT_MAP = {
    "VV-R": "VV",
    "VV-I": "VV",
    "VA-R": "VA",
    "VA-I": "VA",
    "VX-R": "VX",
    "VX-I": "VX",
    "XSA-R": "XSA",
    "XSA-I": "XSA",
    "SSO": "SS",
    "SSC": "SS",
    "XSM": "XSA",
    "Z_CODA": "UNK",
    "Z_SIOT": "UNK",
    "UN": "UNK",
    "MMA": "MM",
    "MMD": "MM",
    "MMN": "MM",
}

PREFIX_MAP = [("W_", "SW")]
USER_TAGS = {f"USER{i}" for i in range(5)}


def map_pos(tag: str) -> str:
    if tag in DIRECT_MAP:
        return DIRECT_MAP[tag]
    if tag in USER_TAGS:
        return "NNP"
    for prefix, target in PREFIX_MAP:
        if tag.startswith(prefix):
            return target
    return "UNK"


def run_pos_mapping(*, cfg: dict[str, Any], run_context: RunContext) -> None:
    logger = logging.getLogger("kmwe")
    outputs_dir = run_context.outputs_dir
    outputs_dir.mkdir(parents=True, exist_ok=True)

    logger.info("stage=pos_mapping 시작")

    mapping_table = {
        "direct_map": DIRECT_MAP,
        "prefix_map": [{"prefix": prefix, "target": target} for prefix, target in PREFIX_MAP],
        "range_map": {"USER0-USER4": "NNP"},
        "fallback": "UNK",
        "generated_at": datetime.now().astimezone().isoformat(),
    }

    report = {
        "n_direct_rules": len(DIRECT_MAP),
        "n_prefix_rules": len(PREFIX_MAP),
        "n_special_rules": 1,
        "examples": {
            "direct_map": list(DIRECT_MAP.items())[:2],
            "prefix_map": [{"input": f"{PREFIX_MAP[0][0]}EX", "output": PREFIX_MAP[0][1]}] if PREFIX_MAP else [],
            "special_rules": [{"input": "USER0", "output": "NNP"}],
        },
    }

    mapping_path = outputs_dir / "pos_mapping.json"
    report_path = outputs_dir / "pos_mapping_report.json"

    write_json(mapping_path, mapping_table, indent=2)
    write_json(report_path, report, indent=2)

    logger.info("stage=pos_mapping 완료: %s", mapping_path)
