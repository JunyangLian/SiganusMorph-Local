"""Prepare human-friendly outlier review materials for v0.6.4 batch measurement."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


BASE = PROJECT_ROOT / "results" / "batch_measurement_v0.6.4_stable"
OUT = BASE / "outlier_review"
PREVIEW_SRC = BASE / "measurement_previews"
PREVIEW_OUT = OUT / "previews"


def _json_loads(text: Any) -> dict[str, Any]:
    if not isinstance(text, str) or not text.strip():
        return {}
    try:
        data = json.loads(text)
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _metric_group(affected_metrics: str) -> str:
    metrics = str(affected_metrics)
    if any(token in metrics for token in ("TL_final", "SL_final", "TL_curve", "SL_curve")):
        return "TL_final"
    if "caudal_peduncle" in metrics:
        return "caudal_peduncle"
    if "head_length" in metrics or "snout_length" in metrics:
        return "head_length"
    return "other"


def _priority(metric_group: str, repeated: bool, outlier_type: str) -> str:
    if repeated or metric_group in {"TL_final", "caudal_peduncle"}:
        return "high"
    if metric_group == "head_length":
        return "medium"
    if "ratio" in outlier_type:
        return "medium"
    return "low"


def _possible_reason(row: dict[str, Any], ratio_values: dict[str, Any], repeated_count: int) -> str:
    group = row["metric_group"]
    metrics = str(row.get("affected_metrics", ""))
    if group == "TL_final":
        if repeated_count > 1:
            return "same specimen has repeated length outliers; likely true large/small specimen or specimen-level scale/label consistency issue"
        return "overall fish length is outside batch distribution; verify P1, P5, P7U/P7L/P7V and board scale"
    if group == "caudal_peduncle":
        return "tail peduncle metric outside distribution; verify P4, P5, P10/P11 and caudal-base placement"
    if group == "head_length":
        return "head/snout metric outside distribution; verify P1, P2, P3 and whether head landmark definition is consistent"
    if "curvature" in metrics:
        return "curvature value flagged statistically; practical curvature threshold may still be acceptable"
    if ratio_values:
        return "ratio metric outside expected range; compare with same-specimen images and preview"
    return "single-metric distribution outlier; may be biological variation or a landmark inconsistency"


def _recommendation(row: dict[str, Any], repeated: bool) -> str:
    group = row["metric_group"]
    image_name = str(row["image_name"])
    if image_name == "real_037.png":
        return "keep excluded; do not use for batch measurement"
    if group == "TL_final":
        return "open Review Queue or preview; check P1/P5/P7U/P7L/P7V before formal analysis"
    if group == "caudal_peduncle":
        return "review tail landmarks P4/P5/P10/P11; compare with same specimen if repeated"
    if group == "head_length":
        return "review P1/P2/P3 definitions; likely landmark consistency check"
    if repeated:
        return "review specimen-level consistency across all images before deciding whether it is biological"
    return "inspect preview; do not auto-exclude"


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("arial.ttf", size=size)
    except Exception:
        return ImageFont.load_default()


def _make_contact_sheet(review: pd.DataFrame) -> Path:
    images = []
    thumb_w, thumb_h = 420, 280
    label_h = 90
    for record in review.to_dict("records"):
        preview = Path(record["preview_path"])
        if not preview.is_absolute():
            preview = PROJECT_ROOT / preview
        if not preview.exists():
            continue
        img = Image.open(preview).convert("RGB")
        img.thumbnail((thumb_w, thumb_h), Image.Resampling.LANCZOS)
        tile = Image.new("RGB", (thumb_w, thumb_h + label_h), (245, 245, 245))
        x = (thumb_w - img.width) // 2
        tile.paste(img, (x, 0))
        draw = ImageDraw.Draw(tile)
        draw.rectangle((0, thumb_h, thumb_w, thumb_h + label_h), fill=(0, 0, 0))
        lines = [
            f"{record['image_name']}  {record['specimen_id']}",
            str(record["affected_metrics"])[:62],
            f"{record['outlier_type']} | {record['source_set']}",
        ]
        y = thumb_h + 6
        for line in lines:
            draw.text((8, y), line, fill=(255, 255, 255), font=_font(14))
            y += 24
        images.append(tile)
    cols = 3
    rows = (len(images) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * thumb_w, max(1, rows) * (thumb_h + label_h)), (230, 230, 230))
    for idx, tile in enumerate(images):
        x = (idx % cols) * thumb_w
        y = (idx // cols) * (thumb_h + label_h)
        sheet.paste(tile, (x, y))
    path = OUT / "outlier_contact_sheet_by_metric.png"
    sheet.save(path)
    return path


def _prepare_review() -> pd.DataFrame:
    OUT.mkdir(parents=True, exist_ok=True)
    PREVIEW_OUT.mkdir(parents=True, exist_ok=True)
    outliers = pd.read_csv(BASE / "measurement_outlier_cases.csv", dtype=str, keep_default_na=False)
    ratios = pd.read_csv(BASE / "batch_measurement_ratios.csv", dtype=str, keep_default_na=False)
    measurements = pd.read_csv(BASE / "batch_measurements.csv", dtype=str, keep_default_na=False)
    ratio_lookup = ratios.set_index("image_name").to_dict("index")
    specimen_to_images = (
        measurements.groupby("specimen_id")["image_name"]
        .apply(lambda s: sorted(set(str(v) for v in s)))
        .to_dict()
    )
    repeated_counts = outliers.groupby("specimen_id")["image_name"].nunique().to_dict()

    rows = []
    for record in outliers.to_dict("records"):
        image_name = str(record["image_name"])
        specimen_id = str(record["specimen_id"])
        group = _metric_group(record.get("affected_metrics", ""))
        src_preview = PREVIEW_SRC / f"{Path(image_name).stem}_measurement_preview.png"
        dst_preview = PREVIEW_OUT / src_preview.name
        if src_preview.exists():
            shutil.copy2(src_preview, dst_preview)
        same_specimen = [img for img in specimen_to_images.get(specimen_id, []) if img != image_name]
        ratio = ratio_lookup.get(image_name, {})
        ratio_values = {
            key: ratio.get(key, "")
            for key in [
                "TL_SL_ratio",
                "body_depth_SL_ratio",
                "head_length_SL_ratio",
                "snout_length_head_length_ratio",
                "caudal_peduncle_length_SL_ratio",
                "caudal_peduncle_depth_SL_ratio",
                "curvature_index_selected",
            ]
            if str(ratio.get(key, "")).strip()
        }
        repeated_count = int(repeated_counts.get(specimen_id, 0))
        repeated = repeated_count > 1
        enriched = {
            **record,
            "metric_group": group,
            "ratio_values": json.dumps(ratio_values, ensure_ascii=False),
            "preview_path": str(dst_preview),
            "same_specimen_other_images": ";".join(same_specimen),
            "possible_reason": "",
            "review_recommendation": "",
            "manual_review_priority": "",
            "repeated_specimen_outlier": bool(repeated),
            "repeated_specimen_count": repeated_count,
            "notes": record.get("notes", ""),
        }
        enriched["possible_reason"] = _possible_reason(enriched, ratio_values, repeated_count)
        enriched["review_recommendation"] = _recommendation(enriched, repeated)
        enriched["manual_review_priority"] = _priority(group, repeated, str(record.get("outlier_type", "")))
        rows.append(enriched)

    review = pd.DataFrame(rows)
    order = {"TL_final": 0, "caudal_peduncle": 1, "head_length": 2, "other": 3}
    priority_order = {"high": 0, "medium": 1, "low": 2}
    review["_group_order"] = review["metric_group"].map(order).fillna(9)
    review["_priority_order"] = review["manual_review_priority"].map(priority_order).fillna(9)
    review = review.sort_values(["_group_order", "_priority_order", "specimen_id", "image_name"]).drop(columns=["_group_order", "_priority_order"])
    columns = [
        "image_name",
        "specimen_id",
        "source_set",
        "metric_group",
        "affected_metrics",
        "outlier_type",
        "metric_values",
        "ratio_values",
        "preview_path",
        "same_specimen_other_images",
        "repeated_specimen_outlier",
        "repeated_specimen_count",
        "possible_reason",
        "review_recommendation",
        "manual_review_priority",
        "notes",
    ]
    review = review.loc[:, columns]
    review.to_csv(OUT / "outlier_grouped_review.csv", index=False, encoding="utf-8-sig")
    return review


def _write_readme(review: pd.DataFrame, contact_sheet: Path) -> None:
    group_counts = review["metric_group"].value_counts().to_dict()
    repeated = (
        review.loc[review["repeated_specimen_outlier"].astype(str).str.lower().isin(["true", "1"]), ["specimen_id", "repeated_specimen_count"]]
        .drop_duplicates()
        .sort_values(["repeated_specimen_count", "specimen_id"], ascending=[False, True])
    )
    high = review[review["manual_review_priority"].eq("high")]
    likely_bio = review[
        review["possible_reason"].str.contains("true large/small specimen|biological", case=False, na=False)
    ]
    text = f"""# v0.6.4 Measurement Outlier Review

## Summary

- Outlier total: {len(review)}
- Contact sheet: `{contact_sheet}`
- Copied previews: `previews/`

## Counts by Metric Group

```json
{json.dumps(group_counts, ensure_ascii=False, indent=2)}
```

## Repeated Specimen Outliers

{repeated.to_markdown(index=False) if not repeated.empty else 'No repeated specimen outliers.'}

Special attention:
- `fish_21`: repeated TL/SL length outliers; check whether these are true large-size images or consistency/scale issues.
- `fish_23`: repeated caudal peduncle length outliers; review P4/P5 placement and tail-base consistency.

## Open First in Review Queue

{high[['image_name', 'specimen_id', 'metric_group', 'affected_metrics', 'review_recommendation']].to_markdown(index=False) if not high.empty else 'No high-priority cases.'}

## Possible True Biological Extremes

{likely_bio[['image_name', 'specimen_id', 'affected_metrics', 'possible_reason']].to_markdown(index=False) if not likely_bio.empty else 'None flagged automatically; inspect repeated specimen cases manually.'}

## Notes

- No images were automatically excluded.
- No measurements were modified.
- No corrected_keypoints were touched.
- Use `outlier_grouped_review.csv` for row-level review and `outlier_contact_sheet_by_metric.png` for fast visual triage.
"""
    (OUT / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    review = _prepare_review()
    contact = _make_contact_sheet(review)
    _write_readme(review, contact)
    print("Prepared v0.6.4 outlier review materials.")
    print(f"Outlier total: {len(review)}")
    print(f"Groups: {review['metric_group'].value_counts().to_dict()}")
    print(f"Repeated specimen outliers: {int(review['repeated_specimen_outlier'].sum())}")
    print(f"Output: {OUT}")


if __name__ == "__main__":
    main()
