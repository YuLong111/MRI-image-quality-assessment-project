from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


def parse_b_value(b_value: Any) -> Optional[int]:
    """Convert b-value tokens like '1k4', '1k6', '2k' to integers.

    - '1k4' -> 1400
    - '1k6' -> 1600
    - '2k'  -> 2000

    Returns None if parsing fails.
    """
    if b_value is None:
        return None
    s = str(b_value).strip().lower()
    # common encodings: 1k4, 1k6, 2k
    if s.endswith("k"):
        try:
            return int(float(s[:-1]) * 1000)
        except ValueError:
            return None
    if "k" in s:
        left, right = s.split("k", 1)
        try:
            return int(left) * 1000 + int(right) * 100
        except ValueError:
            return None
    # maybe already numeric
    try:
        return int(float(s))
    except ValueError:
        return None


def build_datalist_from_xlsx(
    xlsx_path: str | Path,
    image_dir: str | Path,
    out_json: str | Path,
    image_suffix: str = ".nii.gz",
    label_col: str = "score",
    id_col: str = "case_id",
) -> List[Dict[str, Any]]:
    """Build a MONAI-style datalist JSON from a spreadsheet.

    The spreadsheet you provided has: case_id, b_value, score.

    We will:
    - match each case_id to an image file in image_dir
    - store:
        {"id": case_id, "image": <path>, "b_value": <int or None>, "label_bin": <0/1>, "label_score": <int?>}

    NOTE: if your `score` is raw PI-QUAL (0–5), then:
    - label_score = score
    - label_bin = 1 if score>=4 else 0

    If your `score` is already binary, you can still use label_bin=score.
    """
    xlsx_path = Path(xlsx_path)
    image_dir = Path(image_dir)
    out_json = Path(out_json)

    df = pd.read_excel(xlsx_path)
    required = {id_col, label_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Spreadsheet missing columns: {sorted(missing)}")

    items: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        case_id = str(row[id_col]).strip()
        img_path = image_dir / f"{case_id}{image_suffix}"
        if not img_path.exists():
            # also try exact case_id if it already contains suffix
            img_path2 = image_dir / case_id
            if img_path2.exists():
                img_path = img_path2
            else:
                # skip missing files, but keep track in logs upstream
                continue

        score = row[label_col]
        try:
            score_int = int(score)
        except Exception:
            # if score is float or string
            try:
                score_int = int(float(score))
            except Exception:
                score_int = 0

        # Heuristic: if score values are only 0/1 in your sheet, treat as binary.
        # Otherwise treat as PI-QUAL 0..5.
        # We'll still store both fields.
        b_val = parse_b_value(row.get("b_value"))

        item: Dict[str, Any] = {
            "id": case_id,
            "image": str(img_path.resolve()),
            "b_value": b_val,
        }

        # Determine label fields
        if score_int in (0, 1) and df[label_col].dropna().astype(int).isin([0, 1]).all():
            item["label_bin"] = score_int
            item["label_score"] = None
        else:
            item["label_score"] = score_int
            item["label_bin"] = 1 if score_int >= 4 else 0

        items.append(item)

    out_json.parent.mkdir(parents=True, exist_ok=True)
    import json

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(items, f, indent=2)

    return items
