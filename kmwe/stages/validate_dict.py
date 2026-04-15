from __future__ import annotations

import ast
import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Optional

import pandas as pd

from kmwe.core.run_context import RunContext
from kmwe.utils.jsonio import write_json

SHEET1_REQUIRED_KEYS = [
    "e_id",
    "canonical_form",
    "group",
    "polyset_id",
    "spacing_policy",
    "disconti_allowed",
    "e_comp_surf",
    "e_comp_id",
    "default_confidence",
    "detect_ruleset_id",
    "verify_ruleset_id",
    "context_positive_ruleset_id",
    "context_negative_ruleset_id",
]
SHEET1_OPTIONAL_KEYS = [
    "gloss",
    "pragmatics",
    "disambiguation_hint",
    "woo_sense_exists",
    "woo_entry",
    "woo_entry_sense",
]


@dataclass
class Issue:
    severity: str
    issue_code: str
    sheet: str
    row: Optional[int]
    column: Optional[str]
    message: str
    value: Any | None = None
    suggested_fix: dict[str, Any] | None = None
    location: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "severity": self.severity,
            "issue_code": self.issue_code,
            "sheet": self.sheet,
            "row": self.row,
            "column": self.column,
            "message": self.message,
            "suggested_fix": self.suggested_fix,
            "value": self.value,
        }
        if self.location is not None:
            payload["location"] = self.location
        return payload


RE_EID = re.compile(r"^(ept|ece|efe|edc|edf|epf|eae)\d{3}$")
RE_WOO_SENSE = re.compile(r"^\d{3}(;\d{3})*$")
RE_COMP_LIST = re.compile(r"^c\d+(;c\d+)*$")

ENUMS = {
    "sheet1.group": {"a", "b"},
    "sheet1.spacing_policy": {"st", "ls", "nrm"},
    "sheet1.disconti_allowed": {"y", "n"},
    "sheet1.woo_sense_exists": {"y", "n", "un"},
    "sheet2.is_required": {"y", "n"},
    "sheet2.order_policy": {"fx", "fl"},
    "sheet3.scope": {"all", "train", "infer"},
    "sheet3.stage": {"detect", "verify", "context"},
    "sheet3.rule_type": {
        "surface_regex",
        "pos_seq",
        "morph_check",
        "context_pos_regex",
        "context_neg_regex",
    },
    "sheet3.engine": {"re", "posdsl", "json", "python"},
    "sheet3.target": {"raw_sentence", "morph_tokens", "token_window"},
    "sheet4.example_role": {"pos", "dispos", "neg", "conf"},
    "gold.split": {"train", "dev", "test"},
    "gold.pattern_type": {"conti", "disconti"},
    "gold.gold_example_role": {
        "pos_conti",
        "pos_disconti",
        "neg_target_absent",
        "neg_confusable",
        "neg_boundary",
    },
}


def _suggested_fix(action: str, hint: str, example: Any | None = None) -> dict[str, Any]:
    return {"action": action, "hint": hint, "example": example}


def _location(**kwargs: Any) -> dict[str, Any] | None:
    payload = {key: value for key, value in kwargs.items() if value is not None}
    return payload or None


def _normalize_span_text(value: str) -> str:
    if value is None:
        return ""
    compact = re.sub(r"\s+", "", value)
    compact = re.sub(r"[,;|\u00b7\u2026]", "", compact)
    return compact


def run_validate_dict(cfg: dict[str, Any], run_context: RunContext) -> None:
    logger = logging.getLogger("kmwe")
    outputs_dir = run_context.outputs_dir
    outputs_dir.mkdir(parents=True, exist_ok=True)

    dict_xlsx = Path(cfg["paths"]["dict_xlsx"])
    gold_xlsx_value = cfg["paths"].get("gold_xlsx")
    gold_xlsx = Path(gold_xlsx_value) if gold_xlsx_value else None

    validation_cfg = cfg.get("dict", {}).get("validation", {})
    report_version = int(validation_cfg.get("report_version", 1))
    include_suggested_fix = bool(validation_cfg.get("include_suggested_fix", True))
    include_by_issue_code = bool(validation_cfg.get("include_by_issue_code", True))
    severity_levels = validation_cfg.get("severity_levels", ["ERROR", "WARNING", "INFO"])
    llm_cfg = validation_cfg.get("llm_examples", {})
    span_text_mismatch_severity = llm_cfg.get("span_text_mismatch_severity", "WARNING")
    if span_text_mismatch_severity not in {"ERROR", "WARNING", "INFO"}:
        span_text_mismatch_severity = "WARNING"
    try:
        max_pos_examples_per_eid = int(llm_cfg.get("max_pos_examples_per_eid", 4))
    except Exception:
        max_pos_examples_per_eid = 4
    if max_pos_examples_per_eid < 1:
        max_pos_examples_per_eid = 1

    issues: list[Issue] = []
    warnings: list[Issue] = []
    infos: list[Issue] = []

    def add_issue(
        severity: str,
        issue_code: str | None,
        sheet: str,
        row: int | None,
        column: str | None,
        message: str,
        value: Any | None = None,
        suggested_fix: dict[str, Any] | None = None,
        location: dict[str, Any] | None = None,
    ) -> None:
        code = issue_code or "VALIDATION_GENERIC"
        fix = suggested_fix if include_suggested_fix else None
        issue = Issue(
            severity=severity,
            issue_code=code,
            sheet=sheet,
            row=row,
            column=column,
            message=message,
            value=value,
            suggested_fix=fix,
            location=location,
        )
        if severity == "ERROR":
            issues.append(issue)
        elif severity == "WARNING":
            warnings.append(issue)
        else:
            infos.append(issue)

    logger.info("validate_dict 시작: dict_xlsx=%s gold_xlsx=%s", dict_xlsx, gold_xlsx)

    dict_bundle: dict[str, Any] = {
        "meta": {
            "dict_xlsx": str(dict_xlsx),
            "loaded_at": datetime.now().astimezone().isoformat(),
            "schema_version": "v1",
        },
        "expredict": [],
        "components": [],
        "rules": [],
        "llm_examples": [],
    }

    dict_frames = _load_dict_xlsx(dict_xlsx, add_issue)
    if dict_frames is not None:
        sheet1_rows, sheet1_meta = _validate_sheet1(dict_frames["expredict"], add_issue)
        sheet2_rows = _validate_sheet2(dict_frames["components"], sheet1_meta, sheet1_rows, add_issue)
        sheet3_rows, sheet3_rulesets = _validate_sheet3(dict_frames["rules"], sheet1_meta, add_issue)
        sheet4_rows = _validate_sheet4(
            dict_frames["llm_examples"],
            sheet1_meta,
            add_issue,
            span_text_mismatch_severity,
            max_pos_examples_per_eid,
        )
        _validate_ruleset_pointers(sheet1_rows, sheet3_rulesets, add_issue)

        dict_bundle["expredict"] = sheet1_rows
        dict_bundle["components"] = sheet2_rows
        dict_bundle["rules"] = sheet3_rows
        dict_bundle["llm_examples"] = sheet4_rows
        dict_bundle["expredict_map"] = {
            row.get("e_id"): row for row in sheet1_rows if row.get("e_id")
        }

    gold_requires_span = bool(cfg.get("gold", {}).get("neg_confusable_requires_span", False))
    gold_report = _validate_gold_xlsx(gold_xlsx, add_issue, gold_requires_span)

    by_issue_code: dict[str, int] = {}
    if include_by_issue_code:
        for issue in issues + warnings + infos:
            by_issue_code[issue.issue_code] = by_issue_code.get(issue.issue_code, 0) + 1

    report = {
        "report_type": "validate_dict",
        "report_version": report_version,
        "created_at": datetime.now().astimezone().isoformat(),
        "project_root": cfg["paths"]["project_root"],
        "dict_xlsx": str(dict_xlsx),
        "gold_xlsx": str(gold_xlsx) if gold_xlsx else None,
        "status": "failed" if issues else "ok",
        "issues": [issue.to_dict() for issue in issues + warnings + infos],
        "stats": {
            "dict": {
                "sheet1_rows": len(dict_bundle["expredict"]),
                "sheet2_rows": len(dict_bundle["components"]),
                "sheet3_rows": len(dict_bundle["rules"]),
                "sheet4_rows": len(dict_bundle["llm_examples"]),
            },
            "gold": gold_report.get("stats", {}) if gold_report else {},
        },
        "summary": {
            "total_errors": len(issues),
            "total_warnings": len(warnings),
            "total_infos": len(infos),
            "by_issue_code": by_issue_code if include_by_issue_code else {},
            "severity_levels": severity_levels,
        },
        "gold_schema": gold_report,
    }

    report_path = outputs_dir / "validate_dict_report.json"
    write_json(report_path, report, indent=2)

    if issues:
        raise RuntimeError(f"validate_dict 실패: ERROR {len(issues)}건")

    bundle_path = outputs_dir / "dict_bundle.json"
    write_json(bundle_path, dict_bundle, indent=2)
    logger.info("validate_dict 완료: report=%s bundle=%s", report_path, bundle_path)


def _load_dict_xlsx(dict_xlsx: Path, add_issue) -> dict[str, pd.DataFrame] | None:
    if not dict_xlsx.exists():
        add_issue(
            "ERROR",
            "FILE_NOT_FOUND",
            "dict_xlsx",
            None,
            None,
            "dict_xlsx 파일이 존재하지 않습니다.",
            str(dict_xlsx),
            suggested_fix=_suggested_fix("refer_docs", "paths.dict_xlsx 경로를 올바르게 지정하세요.", str(dict_xlsx)),
        )
        return None
    try:
        frames = pd.read_excel(
            dict_xlsx,
            sheet_name=["expredict", "components", "rules", "llm_examples"],
            dtype=object,
            engine="openpyxl",
        )
    except Exception as exc:
        add_issue(
            "ERROR",
            "FILE_READ_FAIL",
            "dict_xlsx",
            None,
            None,
            f"dict_xlsx 로드 실패: {exc}",
            str(dict_xlsx),
            suggested_fix=_suggested_fix("refer_docs", "엑셀 파일 포맷(.xlsx)과 시트명을 확인하세요.", "expredict"),
        )
        return None
    return frames


def _load_gold_xlsx(gold_xlsx: Path | None, add_issue) -> pd.DataFrame | None:
    if gold_xlsx is None:
        add_issue(
            "ERROR",
            "FILE_NOT_FOUND",
            "gold_xlsx",
            None,
            None,
            "gold_xlsx가 설정되지 않았습니다.",
            None,
            suggested_fix=_suggested_fix("fill_required", "paths.gold_xlsx를 설정하세요.", "data/gold.xlsx"),
        )
        return None
    if not gold_xlsx.exists():
        add_issue(
            "ERROR",
            "FILE_NOT_FOUND",
            "gold_xlsx",
            None,
            None,
            "gold_xlsx 파일이 존재하지 않습니다.",
            str(gold_xlsx),
            suggested_fix=_suggested_fix("refer_docs", "paths.gold_xlsx 경로를 올바르게 지정하세요.", str(gold_xlsx)),
        )
        return None
    try:
        frame = pd.read_excel(gold_xlsx, sheet_name=0, dtype=object, engine="openpyxl")
    except Exception as exc:
        add_issue(
            "ERROR",
            "FILE_READ_FAIL",
            "gold_xlsx",
            None,
            None,
            f"gold_xlsx 로드 실패: {exc}",
            str(gold_xlsx),
            suggested_fix=_suggested_fix("refer_docs", "엑셀 파일 포맷(.xlsx)과 시트를 확인하세요.", "gold"),
        )
        return None
    return frame


def _normalize_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, str):
        trimmed = value.strip()
        return trimmed if trimmed != "" else None
    return value


def _row_number(idx: int) -> int:
    return idx + 2


def _require_columns(frame: pd.DataFrame, required: Iterable[str], sheet: str, add_issue) -> bool:
    missing = [col for col in required if col not in frame.columns]
    for col in missing:
        add_issue(
            "ERROR",
            "REQUIRED_COLUMN_MISSING",
            sheet,
            None,
            col,
            "필수 컬럼이 없습니다.",
            suggested_fix=_suggested_fix("fill_required", "엑셀에 필수 컬럼을 추가하세요.", col),
        )
    return len(missing) == 0


def _get_str(row: dict[str, Any], key: str) -> str | None:
    value = row.get(key)
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return str(value)


def _parse_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and value.strip() != "":
        try:
            return int(value)
        except ValueError:
            return None
    return None


def _parse_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if value in (0, 1):
            return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("true", "1"):
            return True
        if lowered in ("false", "0"):
            return False
    return None


def _parse_span_segments(value: Any) -> list[tuple[int, int]] | None:
    if value is None:
        return None
    if isinstance(value, list):
        raw = value
    elif isinstance(value, str):
        if value.strip() == "":
            return None
        try:
            raw = json.loads(value)
        except Exception:
            raw = ast.literal_eval(value)
    else:
        return None

    if not isinstance(raw, list):
        return None

    segments: list[tuple[int, int]] = []
    for item in raw:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            return None
        start = _parse_int(item[0])
        end = _parse_int(item[1])
        if start is None or end is None:
            return None
        segments.append((start, end))
    return segments


def _validate_segments(
    segments: list[tuple[int, int]],
    raw_sentence: str,
    sheet: str,
    row: int,
    add_issue,
) -> None:
    if not segments:
        add_issue(
            "ERROR",
            "SPAN_SEGMENTS_EMPTY",
            sheet,
            row,
            "span_segments",
            "span_segments가 비어있습니다.",
            suggested_fix=_suggested_fix("fill_required", "span_segments를 입력하세요.", "[(0,3)]"),
        )
        return
    sorted_segments = sorted(segments, key=lambda seg: (seg[0], seg[1]))
    last_end = -1
    for start, end in sorted_segments:
        if start < 0 or end <= start or end > len(raw_sentence):
            add_issue(
                "ERROR",
                "SPAN_SEGMENTS_OOB",
                sheet,
                row,
                "span_segments",
                "span_segments 범위가 유효하지 않습니다.",
                {"segment": [start, end], "len": len(raw_sentence)},
                suggested_fix=_suggested_fix("fix_span", "raw_sentence 길이 내로 범위를 조정하세요.", "(0, len(raw_sentence))"),
            )
        if start < last_end:
            add_issue(
                "ERROR",
                "SPAN_SEGMENTS_OVERLAP",
                sheet,
                row,
                "span_segments",
                "span_segments가 overlap 됩니다.",
                {"segment": [start, end]},
                suggested_fix=_suggested_fix("fix_span", "segment 간 겹침이 없도록 수정하세요.", "[(0,3),(5,7)]"),
            )
        last_end = max(last_end, end)


def _validate_sheet1(frame: pd.DataFrame, add_issue):
    sheet = "expredict"
    required = [
        "e_id",
        "group",
        "spacing_policy",
        "disconti_allowed",
        "woo_sense_exists",
        "e_comp_id",
        "default_confidence",
        "detect_ruleset_id",
        "verify_ruleset_id",
        "context_positive_ruleset_id",
        "context_negative_ruleset_id",
        "polyset_id",
        "woo_entry_sense",
    ]
    if not _require_columns(frame, required, sheet, add_issue):
        return [], {"e_ids": set(), "comp_map": {}, "comp_sets": {}}

    seen_eids: set[str] = set()
    comp_map: dict[str, set[str]] = {}
    rows: list[dict[str, Any]] = []

    for idx, raw_row in frame.iterrows():
        row = {col: _normalize_value(raw_row.get(col)) for col in frame.columns}
        row = _ensure_sheet1_keys(row)
        row_no = _row_number(idx)
        e_id = _get_str(row, "e_id")
        if not e_id:
            add_issue(
                "ERROR",
                "REQUIRED_VALUE_MISSING",
                sheet,
                row_no,
                "e_id",
                "e_id가 비어있습니다.",
                suggested_fix=_suggested_fix("fill_required", "e_id 값을 입력하세요.", "ept001"),
            )
        elif e_id in seen_eids:
            add_issue(
                "ERROR",
                "DUPLICATE_EID",
                sheet,
                row_no,
                "e_id",
                "e_id가 중복되었습니다.",
                e_id,
                suggested_fix=_suggested_fix("rename_value", "중복되지 않도록 e_id를 수정하세요.", "ept002"),
            )
        else:
            seen_eids.add(e_id)
            if not RE_EID.match(e_id):
                add_issue(
                    "ERROR",
                    "EID_FORMAT_INVALID",
                    sheet,
                    row_no,
                    "e_id",
                    "e_id 형식이 올바르지 않습니다.",
                    e_id,
                    suggested_fix=_suggested_fix("rename_value", "e_id 형식을 맞추세요.", "ept001"),
                )

        _check_enum(row, sheet, row_no, "group", "sheet1.group", add_issue)
        _check_enum(row, sheet, row_no, "spacing_policy", "sheet1.spacing_policy", add_issue)
        _check_enum(row, sheet, row_no, "disconti_allowed", "sheet1.disconti_allowed", add_issue)
        _check_enum(row, sheet, row_no, "woo_sense_exists", "sheet1.woo_sense_exists", add_issue)

        group_value = (row.get("group") or "").lower()
        polyset_id = _get_str(row, "polyset_id")
        if polyset_id and group_value != "b":
            add_issue(
                "ERROR",
                "POLYSET_GROUP_INVALID",
                sheet,
                row_no,
                "polyset_id",
                "polyset_id가 있는 경우 group은 b여야 합니다.",
                suggested_fix=_suggested_fix("set_enum", "group 값을 b로 수정하세요.", "b"),
            )
        if group_value == "b" and not polyset_id:
            add_issue(
                "ERROR",
                "GROUP_B_POLYSET_REQUIRED",
                sheet,
                row_no,
                "polyset_id",
                "group이 b인 경우 polyset_id가 필수입니다.",
                suggested_fix=_suggested_fix("fill_required", "polyset_id 값을 입력하세요.", "ps001"),
            )

        woo_entry_sense = _get_str(row, "woo_entry_sense")
        if woo_entry_sense and not RE_WOO_SENSE.match(woo_entry_sense):
            add_issue(
                "ERROR",
                "WOO_ENTRY_SENSE_INVALID",
                sheet,
                row_no,
                "woo_entry_sense",
                "woo_entry_sense 형식이 올바르지 않습니다.",
                woo_entry_sense,
                suggested_fix=_suggested_fix("rename_value", "3자리 숫자 세미콜론 형식을 사용하세요.", "001;002"),
            )

        comp_raw = _get_str(row, "e_comp_id")
        if not comp_raw:
            add_issue(
                "ERROR",
                "REQUIRED_VALUE_MISSING",
                sheet,
                row_no,
                "e_comp_id",
                "e_comp_id가 비어있습니다.",
                suggested_fix=_suggested_fix("fill_required", "e_comp_id를 입력하세요.", "c1;c2"),
            )
            comp_ids = set()
        elif not RE_COMP_LIST.match(comp_raw):
            add_issue(
                "ERROR",
                "E_COMP_ID_INVALID",
                sheet,
                row_no,
                "e_comp_id",
                "e_comp_id 형식이 올바르지 않습니다.",
                comp_raw,
                suggested_fix=_suggested_fix("rename_value", "c숫자(;c숫자) 형식을 사용하세요.", "c1;c2"),
            )
            comp_ids = set(comp_raw.split(";"))
        else:
            comp_ids = set(comp_raw.split(";"))
        if e_id:
            comp_map[e_id] = comp_ids

        default_confidence = _parse_int(row.get("default_confidence"))
        if default_confidence is None or not (0 <= default_confidence <= 3):
            add_issue(
                "ERROR",
                "DEFAULT_CONFIDENCE_INVALID",
                sheet,
                row_no,
                "default_confidence",
                "default_confidence는 0~3 정수여야 합니다.",
                row.get("default_confidence"),
                suggested_fix=_suggested_fix("rename_value", "0~3 범위 정수로 수정하세요.", 2),
            )

        rows.append(row)

    return rows, {"e_ids": seen_eids, "comp_map": comp_map}


def _validate_sheet2(
    frame: pd.DataFrame,
    meta: dict[str, Any],
    expredict_rows: list[dict[str, Any]],
    add_issue,
):
    sheet = "components"
    required = [
        "e_id",
        "comp_id",
        "is_required",
        "order_policy",
        "anchor_rank",
        "min_gap_to_next",
        "max_gap_to_next",
        "lemma",
        "pos",
    ]
    if not _require_columns(frame, required, sheet, add_issue):
        return []

    seen_pairs: set[tuple[str, str]] = set()
    comp_counts: dict[str, set[str]] = {}
    rows: list[dict[str, Any]] = []

    disconti_allowed_by_eid = {
        _get_str(row, "e_id"): (_get_str(row, "disconti_allowed") or "").lower()
        for row in expredict_rows
        if _get_str(row, "e_id")
    }

    for idx, raw_row in frame.iterrows():
        row = {col: _normalize_value(raw_row.get(col)) for col in frame.columns}
        row_no = _row_number(idx)
        e_id = _get_str(row, "e_id")
        comp_id = _get_str(row, "comp_id")

        if not e_id:
            add_issue(
                "ERROR",
                "REQUIRED_VALUE_MISSING",
                sheet,
                row_no,
                "e_id",
                "e_id가 비어있습니다.",
                suggested_fix=_suggested_fix("fill_required", "e_id를 입력하세요.", "ept001"),
            )
        elif e_id not in meta["e_ids"]:
            add_issue(
                "ERROR",
                "EID_NOT_FOUND",
                sheet,
                row_no,
                "e_id",
                "sheet1에 존재하지 않는 e_id입니다.",
                e_id,
                suggested_fix=_suggested_fix("refer_docs", "sheet1(expredict)에 e_id를 추가하거나 값을 수정하세요.", "ept001"),
            )

        if not comp_id:
            add_issue(
                "ERROR",
                "REQUIRED_VALUE_MISSING",
                sheet,
                row_no,
                "comp_id",
                "comp_id가 비어있습니다.",
                suggested_fix=_suggested_fix("fill_required", "comp_id를 입력하세요.", "c1"),
            )

        if e_id and comp_id:
            pair = (e_id, comp_id)
            if pair in seen_pairs:
                add_issue(
                    "ERROR",
                    "DUPLICATE_EID_COMP",
                    sheet,
                    row_no,
                    "comp_id",
                    "(e_id, comp_id)가 중복되었습니다.",
                    suggested_fix=_suggested_fix("rename_value", "중복되지 않도록 comp_id를 수정하세요.", "c2"),
                )
            else:
                seen_pairs.add(pair)
            comp_counts.setdefault(e_id, set()).add(comp_id)
            expected = meta["comp_map"].get(e_id, set())
            if expected and comp_id not in expected:
                add_issue(
                    "ERROR",
                    "COMP_ID_NOT_IN_SHEET1",
                    sheet,
                    row_no,
                    "comp_id",
                    "sheet1 e_comp_id에 없는 comp_id입니다.",
                    comp_id,
                    suggested_fix=_suggested_fix("refer_docs", "sheet1 e_comp_id 목록에 포함되도록 수정하세요.", "c1"),
                )

        _check_enum(row, sheet, row_no, "is_required", "sheet2.is_required", add_issue)
        _check_enum(row, sheet, row_no, "order_policy", "sheet2.order_policy", add_issue)

        anchor_rank = _parse_int(row.get("anchor_rank"))
        if anchor_rank is None or anchor_rank < 0:
            add_issue(
                "ERROR",
                "ANCHOR_RANK_INVALID",
                sheet,
                row_no,
                "anchor_rank",
                "anchor_rank는 0 이상의 정수여야 합니다.",
                row.get("anchor_rank"),
                suggested_fix=_suggested_fix("rename_value", "0 이상의 정수로 수정하세요.", 0),
            )

        min_gap = _parse_int(row.get("min_gap_to_next"))
        max_gap = _parse_int(row.get("max_gap_to_next"))
        disconti_allowed = disconti_allowed_by_eid.get(e_id)
        if min_gap is None or max_gap is None:
            if disconti_allowed == "n":
                min_gap = min_gap or 0
                max_gap = max_gap or 0
            else:
                add_issue(
                    "WARNING",
                    "GAP_DEFAULTED",
                    sheet,
                    row_no,
                    "min_gap_to_next",
                    "min/max gap이 비어있어 0으로 간주합니다.",
                    suggested_fix=_suggested_fix("fill_required", "min_gap_to_next/max_gap_to_next를 명시하세요.", 0),
                )
                min_gap = min_gap or 0
                max_gap = max_gap or 0
        if min_gap < 0 or max_gap < 0 or min_gap > max_gap:
            add_issue(
                "ERROR",
                "GAP_RANGE_INVALID",
                sheet,
                row_no,
                "min_gap_to_next",
                "min_gap_to_next <= max_gap_to_next 조건을 위반했습니다.",
                suggested_fix=_suggested_fix("rename_value", "min<=max 조건을 맞추세요.", "0<=1"),
            )
        row["min_gap_to_next"] = min_gap
        row["max_gap_to_next"] = max_gap

        pos = _get_str(row, "pos")
        if not pos:
            add_issue(
                "ERROR",
                "REQUIRED_VALUE_MISSING",
                sheet,
                row_no,
                "pos",
                "pos가 비어있습니다.",
                suggested_fix=_suggested_fix("fill_required", "pos 값을 입력하세요.", "NNG"),
            )
        else:
            parts = [p.strip() for p in pos.split(";") if p.strip()]
            row["pos"] = ";".join(parts)

        lemma = _get_str(row, "lemma")
        if pos and any("ETM" in p for p in pos.split(";")) and lemma:
            add_issue(
                "ERROR",
                "LEMMA_NOT_ALLOWED",
                sheet,
                row_no,
                "lemma",
                "ETM 계열 POS에서는 lemma가 비어있어야 합니다.",
                suggested_fix=_suggested_fix("set_empty", "lemma를 비워주세요.", None),
            )

        rows.append(row)

    for e_id, expected in meta["comp_map"].items():
        actual = comp_counts.get(e_id, set())
        if expected and actual != expected:
            add_issue(
                "ERROR",
                "COMP_ID_SET_MISMATCH",
                sheet,
                None,
                "comp_id",
                "sheet1 e_comp_id와 sheet2 comp_id가 불일치합니다.",
                {"e_id": e_id, "expected": sorted(expected), "actual": sorted(actual)},
                suggested_fix=_suggested_fix("refer_docs", "sheet1 e_comp_id와 sheet2 comp_id를 일치시키세요.", "c1;c2"),
            )

    return rows


def _ensure_sheet1_keys(row: dict[str, Any]) -> dict[str, Any]:
    for key in SHEET1_REQUIRED_KEYS + SHEET1_OPTIONAL_KEYS:
        row.setdefault(key, None)
    return row


def _validate_sheet3(frame: pd.DataFrame, meta: dict[str, Any], add_issue):
    sheet = "rules"
    required = [
        "e_id",
        "ruleset_id",
        "rule_id",
        "scope",
        "stage",
        "rule_type",
        "engine",
        "target",
        "comp_id",
        "priority",
        "hard_fail",
        "confidence_delta",
    ]
    if not _require_columns(frame, required, sheet, add_issue):
        return [], set()

    seen_pairs: set[tuple[str, str]] = set()
    ruleset_ids: set[str] = set()
    rows: list[dict[str, Any]] = []
    context_hard_fail = 0

    for idx, raw_row in frame.iterrows():
        row = {col: _normalize_value(raw_row.get(col)) for col in frame.columns}
        row_no = _row_number(idx)
        e_id = _get_str(row, "e_id")
        if not e_id:
            add_issue(
                "ERROR",
                "REQUIRED_VALUE_MISSING",
                sheet,
                row_no,
                "e_id",
                "e_id가 비어있습니다.",
                suggested_fix=_suggested_fix("fill_required", "e_id를 입력하세요.", "ept001"),
            )
        elif e_id not in meta["e_ids"]:
            add_issue(
                "ERROR",
                "EID_NOT_FOUND",
                sheet,
                row_no,
                "e_id",
                "sheet1에 존재하지 않는 e_id입니다.",
                e_id,
                suggested_fix=_suggested_fix("refer_docs", "sheet1(expredict)에 e_id를 추가하거나 값을 수정하세요.", "ept001"),
            )

        ruleset_id = _get_str(row, "ruleset_id")
        rule_id = _get_str(row, "rule_id")
        if ruleset_id and rule_id:
            key = (ruleset_id, rule_id)
            if key in seen_pairs:
                add_issue(
                    "ERROR",
                    "DUPLICATE_RULE_ID",
                    sheet,
                    row_no,
                    "rule_id",
                    "(ruleset_id, rule_id)가 중복되었습니다.",
                    suggested_fix=_suggested_fix("rename_value", "중복되지 않도록 rule_id를 수정하세요.", "r_ept001_d02"),
                )
            else:
                seen_pairs.add(key)

        if ruleset_id:
            ruleset_ids.add(ruleset_id)
            if e_id and not re.match(rf"^rs_({e_id})_(d|v|cp|cn)\d{{2}}$", ruleset_id):
                if e_id not in ruleset_id:
                    add_issue(
                        "ERROR",
                        "RULESET_EID_MISMATCH",
                        sheet,
                        row_no,
                        "ruleset_id",
                        "ruleset_id에 e_id가 포함되어야 합니다.",
                        ruleset_id,
                        suggested_fix=_suggested_fix("rename_value", "ruleset_id에 e_id를 포함하세요.", f"rs_{e_id}_d01"),
                    )
                else:
                    add_issue(
                        "WARNING",
                        "RULESET_FORMAT_NONSTANDARD",
                        sheet,
                        row_no,
                        "ruleset_id",
                        "ruleset_id 형식이 권장 규칙과 다릅니다.",
                        ruleset_id,
                        suggested_fix=_suggested_fix("refer_docs", "권장 규칙(rs_<e_id>_d01) 형식을 확인하세요.", f"rs_{e_id}_d01"),
                    )

        if rule_id:
            if e_id and not re.match(rf"^r_({e_id})_(d|v|cp|cn)\d{{2}}$", rule_id):
                if e_id not in rule_id:
                    add_issue(
                        "ERROR",
                        "RULE_EID_MISMATCH",
                        sheet,
                        row_no,
                        "rule_id",
                        "rule_id에 e_id가 포함되어야 합니다.",
                        rule_id,
                        suggested_fix=_suggested_fix("rename_value", "rule_id에 e_id를 포함하세요.", f"r_{e_id}_d01"),
                    )
                else:
                    add_issue(
                        "WARNING",
                        "RULE_FORMAT_NONSTANDARD",
                        sheet,
                        row_no,
                        "rule_id",
                        "rule_id 형식이 권장 규칙과 다릅니다.",
                        rule_id,
                        suggested_fix=_suggested_fix("refer_docs", "권장 규칙(r_<e_id>_d01) 형식을 확인하세요.", f"r_{e_id}_d01"),
                    )

        comp_id = _get_str(row, "comp_id")
        if comp_id:
            expected = meta["comp_map"].get(e_id or "", set())
            if expected and comp_id not in expected:
                add_issue(
                    "ERROR",
                    "COMP_ID_NOT_IN_SHEET1",
                    sheet,
                    row_no,
                    "comp_id",
                    "sheet1 e_comp_id에 없는 comp_id입니다.",
                    comp_id,
                    suggested_fix=_suggested_fix("refer_docs", "sheet1 e_comp_id 목록에 포함되도록 수정하세요.", "c1"),
                )

        _check_enum(row, sheet, row_no, "scope", "sheet3.scope", add_issue, issue_code="ENUM_RULES_SCOPE_INVALID")
        _check_enum(row, sheet, row_no, "stage", "sheet3.stage", add_issue)
        _check_enum(row, sheet, row_no, "rule_type", "sheet3.rule_type", add_issue)
        _check_enum(row, sheet, row_no, "engine", "sheet3.engine", add_issue)
        _check_enum(row, sheet, row_no, "target", "sheet3.target", add_issue)

        priority = _parse_int(row.get("priority"))
        if priority is None:
            add_issue(
                "ERROR",
                "PRIORITY_INVALID",
                sheet,
                row_no,
                "priority",
                "priority는 정수여야 합니다.",
                row.get("priority"),
                suggested_fix=_suggested_fix("rename_value", "정수로 수정하세요.", 1),
            )
        row["priority"] = priority

        hard_fail = _parse_bool(row.get("hard_fail"))
        if hard_fail is None:
            add_issue(
                "ERROR",
                "HARD_FAIL_INVALID",
                sheet,
                row_no,
                "hard_fail",
                "hard_fail은 bool로 파싱되어야 합니다.",
                row.get("hard_fail"),
                suggested_fix=_suggested_fix("rename_value", "true/false 또는 1/0으로 입력하세요.", "false"),
            )
        row["hard_fail"] = hard_fail
        if (row.get("stage") or "").lower() == "context" and hard_fail:
            context_hard_fail += 1

        confidence_delta = _parse_int(row.get("confidence_delta"))
        if confidence_delta is None:
            add_issue(
                "ERROR",
                "CONFIDENCE_DELTA_INVALID",
                sheet,
                row_no,
                "confidence_delta",
                "confidence_delta는 정수여야 합니다.",
                row.get("confidence_delta"),
                suggested_fix=_suggested_fix("rename_value", "정수로 수정하세요.", 1),
            )
        row["confidence_delta"] = confidence_delta

        rows.append(row)

    if context_hard_fail > 0:
        add_issue(
            "WARNING",
            "CONTEXT_HARD_FAIL_PRESENT",
            sheet,
            None,
            "hard_fail",
            "context stage의 hard_fail 규칙이 존재합니다.",
            context_hard_fail,
            suggested_fix=_suggested_fix("refer_docs", "context 단계 hard_fail 최소화를 권장합니다.", None),
        )

    return rows, ruleset_ids


def _validate_sheet4(
    frame: pd.DataFrame,
    meta: dict[str, Any],
    add_issue,
    span_text_mismatch_severity: str,
    max_pos_examples_per_eid: int,
):
    sheet = "llm_examples"
    required = [
        "e_id",
        "example_id",
        "instance_id",
        "example_role",
        "raw_sentence",
        "span_segments",
    ]
    if not _require_columns(frame, required, sheet, add_issue):
        return []

    seen_keys: set[tuple[str, str, int]] = set()
    pos_count_by_eid: dict[str, int] = {}
    rows: list[dict[str, Any]] = []

    for idx, raw_row in frame.iterrows():
        row = {col: _normalize_value(raw_row.get(col)) for col in frame.columns}
        row_no = _row_number(idx)
        e_id = _get_str(row, "e_id")
        example_id = _get_str(row, "example_id")
        instance_id = _parse_int(row.get("instance_id"))
        location = _location(example_id=example_id, e_id=e_id, instance_id=instance_id)
        example_role = _get_str(row, "example_role")
        raw_sentence = _get_str(row, "raw_sentence")

        if not e_id:
            add_issue(
                "ERROR",
                "REQUIRED_VALUE_MISSING",
                sheet,
                row_no,
                "e_id",
                "e_id가 비어있습니다.",
                suggested_fix=_suggested_fix("fill_required", "e_id를 입력하세요.", "ept001"),
                location=location,
            )
        elif e_id not in meta["e_ids"]:
            add_issue(
                "ERROR",
                "EID_NOT_FOUND",
                sheet,
                row_no,
                "e_id",
                "sheet1에 존재하지 않는 e_id입니다.",
                e_id,
                suggested_fix=_suggested_fix("refer_docs", "sheet1(expredict)에 e_id를 추가하거나 값을 수정하세요.", "ept001"),
                location=location,
            )

        if not example_id:
            add_issue(
                "ERROR",
                "REQUIRED_VALUE_MISSING",
                sheet,
                row_no,
                "example_id",
                "example_id가 비어있습니다.",
                suggested_fix=_suggested_fix("fill_required", "example_id를 입력하세요.", "ex1"),
                location=location,
            )

        if instance_id is None or instance_id < 1:
            add_issue(
                "ERROR",
                "INSTANCE_ID_INVALID",
                sheet,
                row_no,
                "instance_id",
                "instance_id는 1 이상의 정수여야 합니다.",
                row.get("instance_id"),
                suggested_fix=_suggested_fix("rename_value", "1 이상의 정수로 수정하세요.", 1),
                location=location,
            )

        if example_role:
            _check_enum(row, sheet, row_no, "example_role", "sheet4.example_role", add_issue)

        if example_role == "pos" and e_id:
            new_count = pos_count_by_eid.get(e_id, 0) + 1
            pos_count_by_eid[e_id] = new_count
            if new_count > max_pos_examples_per_eid:
                add_issue(
                    "ERROR",
                    "TOO_MANY_POS_EXAMPLES_PER_EID",
                    sheet,
                    row_no,
                    "example_role",
                    f"e_id별 pos 예문은 최대 {max_pos_examples_per_eid}개까지 허용됩니다.",
                    {"e_id": e_id, "pos_count": new_count, "max_allowed": max_pos_examples_per_eid},
                    suggested_fix=_suggested_fix(
                        "remove_row",
                        f"동일 e_id의 pos 예문 수를 {max_pos_examples_per_eid}개 이하로 조정하세요.",
                        None,
                    ),
                    location=location,
                )

        if example_role == "conf":
            # Backward compatibility: accept either `conf_note` or legacy `note`.
            conf_note = _get_str(row, "conf_note") or _get_str(row, "note")
            if not conf_note:
                add_issue(
                    "ERROR",
                    "CONF_NOTE_REQUIRED",
                    sheet,
                    row_no,
                    "conf_note",
                    "example_role=conf 인 경우 conf_note 또는 note가 필요합니다.",
                    suggested_fix=_suggested_fix("fill_required", "conf_note 또는 note를 입력하세요.", "혼동 의미 설명"),
                    location=location,
                )

        if not raw_sentence:
            add_issue(
                "ERROR",
                "REQUIRED_VALUE_MISSING",
                sheet,
                row_no,
                "raw_sentence",
                "raw_sentence는 비어있으면 안 됩니다.",
                suggested_fix=_suggested_fix("fill_required", "raw_sentence를 입력하세요.", "예문"),
                location=location,
            )

        if e_id and example_id and instance_id is not None:
            key = (e_id, example_id, instance_id)
            if key in seen_keys:
                add_issue(
                    "ERROR",
                    "DUPLICATE_EXAMPLE_ID",
                    sheet,
                    row_no,
                    "example_id",
                    "(e_id, example_id, instance_id)가 중복되었습니다.",
                    suggested_fix=_suggested_fix("rename_value", "example_id 또는 instance_id를 수정하세요.", "ex2"),
                location=location,
                )
            else:
                seen_keys.add(key)

        raw_span_value = row.get("span_segments")
        raw_span_str = str(raw_span_value).strip() if raw_span_value is not None else ""
        segments = _parse_span_segments(raw_span_value)
        is_span_empty = raw_span_value is None or raw_span_str == ""

        if is_span_empty:
            if example_role != "neg":
                add_issue(
                    "ERROR",
                    "LLM_EXAMPLE_MISSING_SPAN",
                    sheet,
                    row_no,
                    "span_segments",
                    "neg가 아닌 예문에서는 span_segments가 필요합니다.",
                    suggested_fix=_suggested_fix("add_span", "span_segments를 입력하세요. 예: [(0,3)]", "[(0,3)]"),
                    location=location,
                )
        else:
            if segments is None:
                add_issue(
                    "ERROR",
                    "SPAN_SEGMENTS_PARSE_FAIL",
                    sheet,
                    row_no,
                    "span_segments",
                    "span_segments 파싱 실패.",
                    raw_span_value,
                    suggested_fix=_suggested_fix("fix_span", "span_segments를 리스트 문자열로 입력하세요.", "[(0,3)]"),
                    location=location,
                )

        if raw_sentence and segments:
            _validate_segments(segments, raw_sentence, sheet, row_no, add_issue)
            span_text = _get_str(row, "span_text") if "span_text" in row else None
            if span_text and span_text.strip() != "":
                joined = "".join(raw_sentence[s:e] for s, e in segments)
                if _normalize_span_text(joined) != _normalize_span_text(span_text):
                    add_issue(
                        span_text_mismatch_severity,
                        "SPAN_TEXT_MISMATCH",
                        sheet,
                        row_no,
                        "span_text",
                        "span_text와 span_segments 추출 결과가 일치하지 않습니다. (공백/구분자 제거 기준)",
                        suggested_fix=_suggested_fix(
                            "fix_span",
                            "정규화(공백/구분자 제거) 기준으로도 불일치합니다. span_text 또는 span_segments를 수정하세요.",
                            joined,
                        ),
                        location=location,
                    )

        row["instance_id"] = instance_id
        row["span_segments"] = segments
        rows.append(row)

    return rows


def _validate_ruleset_pointers(sheet1_rows: list[dict[str, Any]], ruleset_ids: set[str], add_issue) -> None:
    pointer_cols = [
        "detect_ruleset_id",
        "verify_ruleset_id",
        "context_positive_ruleset_id",
        "context_negative_ruleset_id",
    ]
    for idx, row in enumerate(sheet1_rows):
        row_no = _row_number(idx)
        for col in pointer_cols:
            value = _get_str(row, col)
            if value and value not in ruleset_ids:
                add_issue(
                    "ERROR",
                    "RULESET_NOT_FOUND",
                    "expredict",
                    row_no,
                    col,
                    "ruleset_id가 sheet3에 존재하지 않습니다.",
                    value,
                    suggested_fix=_suggested_fix("refer_docs", "sheet3 ruleset_id 목록을 확인하세요.", value),
                )


def _validate_gold_xlsx(gold_xlsx: Path, add_issue, neg_confusable_requires_span: bool) -> dict[str, Any]:
    frame = _load_gold_xlsx(gold_xlsx, add_issue)
    if frame is None:
        return {}

    sheet = "gold"
    required = [
        "e_id",
        "example_id",
        "context_left",
        "target_sentence",
        "context_right",
        "instance_id",
        "split",
        "span_segments",
        "pattern_type",
        "gold_example_role",
        "source",
        "conf_e_id",
        "note",
    ]
    if not _require_columns(frame, required, sheet, add_issue):
        return {"stats": {"rows": 0}}

    seen_match_keys: set[tuple[str, int]] = set()
    rows = 0
    for idx, raw_row in frame.iterrows():
        rows += 1
        row = {col: _normalize_value(raw_row.get(col)) for col in frame.columns}
        row_no = _row_number(idx)

        e_id = _get_str(row, "e_id")
        example_id = _get_str(row, "example_id")
        location = _location(example_id=example_id, e_id=e_id)
        if not example_id:
            add_issue(
                "ERROR",
                "REQUIRED_VALUE_MISSING",
                sheet,
                row_no,
                "example_id",
                "example_id가 비어있습니다.",
                suggested_fix=_suggested_fix("fill_required", "example_id를 입력하세요.", "g001"),
                location=location,
            )
        instance_id = _parse_int(row.get("instance_id"))
        if instance_id is None or instance_id < 1:
            add_issue(
                "ERROR",
                "GOLD_INSTANCE_ID_INVALID",
                sheet,
                row_no,
                "instance_id",
                "instance_id는 1 이상의 정수여야 합니다.",
                row.get("instance_id"),
                suggested_fix=_suggested_fix("rename_value", "1 이상의 정수로 수정하세요.", 1),
                location=location,
            )
        else:
            match_key = (str(example_id).strip(), instance_id)
            if match_key in seen_match_keys:
                add_issue(
                    "ERROR",
                    "DUPLICATE_EXAMPLE_ID",
                    sheet,
                    row_no,
                    "example_id",
                    "example_id는 중복 가능하지만, (example_id, instance_id) 조합은 유일해야 합니다.",
                    suggested_fix=_suggested_fix(
                        "fix_instance_id_or_split_row",
                        "동일 example_id 내 instance_id를 1,2,3...로 유일하게 부여하거나 중복 행을 제거하세요.",
                        "example_id=g0203, instance_id=1 / example_id=g0203, instance_id=2",
                    ),
                    location=location,
                )
            else:
                seen_match_keys.add(match_key)

        _check_enum(row, sheet, row_no, "split", "gold.split", add_issue)
        _check_enum(row, sheet, row_no, "gold_example_role", "gold.gold_example_role", add_issue)

        target_sentence = _get_str(row, "target_sentence")
        if not target_sentence:
            add_issue(
                "ERROR",
                "REQUIRED_VALUE_MISSING",
                sheet,
                row_no,
                "target_sentence",
                "target_sentence가 비어있습니다.",
                suggested_fix=_suggested_fix("fill_required", "target_sentence를 입력하세요.", "예문"),
                location=location,
            )

        raw_span_value = row.get("span_segments")
        segments = _parse_span_segments(raw_span_value)
        pattern_type = _get_str(row, "pattern_type")
        gold_role = _get_str(row, "gold_example_role")

        if raw_span_value is not None and segments is None and str(raw_span_value).strip() != "":
            add_issue(
                "ERROR",
                "SPAN_SEGMENTS_PARSE_FAIL",
                sheet,
                row_no,
                "span_segments",
                "span_segments 파싱 실패.",
                raw_span_value,
                suggested_fix=_suggested_fix("fix_span", "span_segments를 리스트 문자열로 입력하세요.", "[(0,3)]"),
                location=location,
            )

        if gold_role in ("pos_conti", "pos_disconti"):
            if segments is None:
                add_issue(
                    "ERROR",
                    "SPAN_REQUIRED",
                    sheet,
                    row_no,
                    "span_segments",
                    "pos_* 에서는 span_segments가 필수입니다.",
                    suggested_fix=_suggested_fix("fill_required", "span_segments를 입력하세요.", "[(0,3)]"),
                    location=location,
                )
            if pattern_type not in ENUMS["gold.pattern_type"]:
                add_issue(
                    "ERROR",
                    "ENUM_VALUE_INVALID",
                    sheet,
                    row_no,
                    "pattern_type",
                    "pattern_type 값이 conti/disconti가 아니면 허용되지 않은 enum 값입니다.",
                    pattern_type,
                    suggested_fix=_suggested_fix("set_enum", "pattern_type을 conti/disconti로 설정하세요.", "conti"),
                    location=location,
                )

        if gold_role == "neg_target_absent":
            if pattern_type:
                add_issue(
                    "ERROR",
                    "GOLD_NEG_TARGET_ABSENT_HAS_PATTERN_TYPE",
                    sheet,
                    row_no,
                    "pattern_type",
                    "neg_target_absent 에서는 pattern_type이 비어있어야 합니다.",
                    suggested_fix=_suggested_fix("set_empty", "pattern_type을 비우세요.", None),
                location=location,
                )

        if gold_role == "neg_boundary":
            if segments is None:
                add_issue(
                    "ERROR",
                    "GOLD_NEG_BOUNDARY_MISSING_SPAN",
                    sheet,
                    row_no,
                    "span_segments",
                    "neg_boundary 에서는 span_segments가 필요합니다. (잘못된 경계 span 명시)",
                    suggested_fix=_suggested_fix("fill_required", "잘못된 경계 span을 입력하세요.", "[(0,3)]"),
                location=location,
                )
            if pattern_type is None:
                add_issue(
                    "WARNING",
                    "PATTERN_TYPE_EMPTY",
                    sheet,
                    row_no,
                    "pattern_type",
                    "pattern_type이 비어있습니다(자동 산출은 하지 않습니다).",
                    suggested_fix=_suggested_fix("set_enum", "필요하면 pattern_type을 conti/disconti로 설정하세요.", "conti"),
                location=location,
                )
            else:
                if pattern_type not in ENUMS["gold.pattern_type"]:
                    add_issue(
                        "ERROR",
                        "ENUM_VALUE_INVALID",
                        sheet,
                        row_no,
                        "pattern_type",
                        "pattern_type 값이 conti/disconti가 아니면 허용되지 않은 enum 값입니다.",
                        pattern_type,
                        suggested_fix=_suggested_fix("set_enum", "pattern_type을 conti/disconti로 설정하세요.", "conti"),
                    location=location,
                    )
                if segments is None:
                    add_issue(
                        "ERROR",
                        "SPAN_REQUIRED",
                        sheet,
                        row_no,
                        "pattern_type",
                        "pattern_type 검증을 위해 span_segments가 필요합니다.",
                        suggested_fix=_suggested_fix("fill_required", "span_segments를 입력하세요.", "[(0,3)]"),
                    location=location,
                    )

        if gold_role == "neg_confusable":
            conf_e_id = _get_str(row, "conf_e_id")
            if not conf_e_id:
                add_issue(
                    "ERROR",
                    "CONF_EID_REQUIRED",
                    sheet,
                    row_no,
                    "conf_e_id",
                    "neg_confusable은 conf_e_id가 필요합니다.",
                    suggested_fix=_suggested_fix("fill_required", "conf_e_id를 입력하세요.", "ept001"),
                location=location,
                )
            if segments is None:
                if neg_confusable_requires_span:
                    add_issue(
                        "ERROR",
                        "GOLD_NEG_CONFUSABLE_MISSING_SPAN",
                        sheet,
                        row_no,
                        "span_segments",
                        "gold.neg_confusable_requires_span=true 이므로 span_segments가 필수입니다.",
                        suggested_fix=_suggested_fix("fill_required", "confusable span을 입력하세요.", "[(0,3)]"),
                        location=location,
                    )
                else:
                    add_issue(
                        "WARNING",
                        "GOLD_NEG_CONFUSABLE_MISSING_SPAN",
                        sheet,
                        row_no,
                        "span_segments",
                        "neg_confusable 에서는 span_segments가 권장됩니다. 학습에서 confusable 구간(헷갈린 표면형)을 명시하면 후보-span 판별 성능이 좋아집니다.",
                        suggested_fix=_suggested_fix("fill_required", "confusable span을 입력하면 학습 품질이 좋아집니다.", "[(0,3)]"),
                        location=location,
                    )
            if pattern_type is None:
                add_issue(
                    "WARNING",
                    "PATTERN_TYPE_EMPTY",
                    sheet,
                    row_no,
                    "pattern_type",
                    "pattern_type이 비어있습니다(자동 산출은 하지 않습니다).",
                    suggested_fix=_suggested_fix("set_enum", "필요하면 pattern_type을 conti/disconti로 설정하세요.", "conti"),
                location=location,
                )
            else:
                if pattern_type not in ENUMS["gold.pattern_type"]:
                    add_issue(
                        "ERROR",
                        "ENUM_VALUE_INVALID",
                        sheet,
                        row_no,
                        "pattern_type",
                        "pattern_type 값이 conti/disconti가 아니면 허용되지 않은 enum 값입니다.",
                        pattern_type,
                        suggested_fix=_suggested_fix("set_enum", "pattern_type을 conti/disconti로 설정하세요.", "conti"),
                    location=location,
                    )
                if segments is None:
                    add_issue(
                        "ERROR",
                        "SPAN_REQUIRED",
                        sheet,
                        row_no,
                        "pattern_type",
                        "pattern_type 검증을 위해 span_segments가 필요합니다.",
                        suggested_fix=_suggested_fix("fill_required", "span_segments를 입력하세요.", "[(0,3)]"),
                    location=location,
                    )

        if target_sentence and segments:
            _validate_segments(segments, target_sentence, sheet, row_no, add_issue)
            if gold_role == "pos_conti" and len(segments) != 1:
                add_issue(
                    "ERROR",
                    "SPAN_SEGMENT_COUNT_INVALID",
                    sheet,
                    row_no,
                    "span_segments",
                    "pos_conti는 segment 1개여야 합니다.",
                    suggested_fix=_suggested_fix("fix_span", "segment 1개로 수정하세요.", "[(0,3)]"),
                    location=location,
                )
            if gold_role == "pos_disconti" and len(segments) < 2:
                add_issue(
                    "ERROR",
                    "SPAN_SEGMENT_COUNT_INVALID",
                    sheet,
                    row_no,
                    "span_segments",
                    "pos_disconti는 segment 2개 이상이어야 합니다.",
                    suggested_fix=_suggested_fix("fix_span", "segment를 2개 이상 입력하세요.", "[(0,1),(2,3)]"),
                    location=location,
                )
            if gold_role in ("neg_boundary", "neg_confusable", "pos_conti", "pos_disconti"):
                if pattern_type == "conti" and len(segments) != 1:
                    add_issue(
                        "ERROR",
                        "PATTERN_TYPE_MISMATCH",
                        sheet,
                        row_no,
                        "pattern_type",
                        "pattern_type=conti인데 span_segments segment가 2개 이상입니다.",
                        suggested_fix=_suggested_fix("fix_span", "conti면 segment를 1개로 수정하세요.", "[(0,3)]"),
                    location=location,
                    )
                if pattern_type == "disconti" and len(segments) == 1:
                    add_issue(
                        "ERROR",
                        "PATTERN_TYPE_MISMATCH",
                        sheet,
                        row_no,
                        "pattern_type",
                        "pattern_type=disconti인데 span_segments segment가 1개입니다.",
                        suggested_fix=_suggested_fix("fix_span", "disconti면 segment를 2개 이상으로 수정하세요.", "[(0,1),(2,3)]"),
                    location=location,
                    )

    return {"stats": {"rows": rows}}


def _check_enum(
    row: dict[str, Any],
    sheet: str,
    row_no: int,
    column: str,
    enum_key: str,
    add_issue,
    issue_code: str | None = None,
) -> None:
    allowed = ENUMS.get(enum_key)
    if not allowed:
        return
    value = _get_str(row, column)
    if value is None:
        add_issue(
            "ERROR",
            issue_code or "REQUIRED_VALUE_MISSING",
            sheet,
            row_no,
            column,
            "필수 값이 비어있습니다.",
            suggested_fix=_suggested_fix("fill_required", f"{column} 값을 입력하세요.", next(iter(allowed)) if allowed else None),
        )
        return
    if value not in allowed:
        add_issue(
            "ERROR",
            issue_code or "ENUM_VALUE_INVALID",
            sheet,
            row_no,
            column,
            "허용되지 않은 enum 값입니다.",
            value,
            suggested_fix=_suggested_fix("set_enum", f"{column} 값을 허용된 enum으로 수정하세요.", sorted(allowed)),
        )
