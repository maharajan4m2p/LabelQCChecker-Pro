"""
Label QC Checker Pro - Comparison Pipeline
Line-by-line Approval vs Sample comparison with
coordinate-accurate highlighting.
"""

from pathlib import Path
import cv2

from engine.ocr_engine import OCREngine
from engine.field_detector import FieldDetector
from engine.comparison_engine import ComparisonEngine
from engine.logo_checker import LogoChecker
from engine.barcode_checker import BarcodeChecker
from engine.image_highlighter import ImageHighlighter


def make_browser_url(path):
    path = Path(path)
    if path.parent.name.lower() == "uploads":
        return f"/uploads/{path.name}"
    if path.parent.name.lower() == "outputs":
        return f"/outputs/{path.name}"
    return str(path)


def calculate_overall_status(comparator, score, comparison, logo_result, barcode_result):
    c = comparison or {}
    if any(c.get(k, 0) > 0 for k in (
        "mismatched", "missing", "extra",
        "line_mismatched", "line_missing", "line_extra",
    )):
        return "FAIL"
    if str((logo_result or {}).get("status", "")).upper() == "FAIL":
        return "FAIL"
    if str((barcode_result or {}).get("status", "")).upper() == "FAIL":
        return "FAIL"
    return "PASS" if float(score) >= comparator.overall_pass_threshold else "FAIL"


def _safe_image(path):
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Unable to read label image: {path}")
    return image


def create_highlighted_images(
    highlighter,
    approval_path,
    sample_path,
    approval_words,
    sample_words,
    comparison,
):
    approval_image = _safe_image(approval_path)
    sample_image = _safe_image(sample_path)

    diffs = (
        comparison.get("visual_differences", [])
        or comparison.get("line_result", {}).get("visual_differences", [])
        or []
    )

    approval_output = highlighter.create_output_path(
        "approval_highlighted", ".jpg"
    )
    sample_output = highlighter.create_output_path(
        "sample_highlighted", ".jpg"
    )
    side_output = highlighter.create_output_path(
        "approval_sample_side_by_side", ".jpg"
    )

    approval_result = highlighter.highlight_visual_differences(
        approval_image,
        approval_words,
        diffs,
        "approval",
        approval_output,
    )

    sample_result = highlighter.highlight_visual_differences(
        sample_image,
        sample_words,
        diffs,
        "sample",
        sample_output,
    )

    # The side-by-side image uses the already highlighted images,
    # guaranteeing that both panels show exactly the same annotations.
    approval_highlighted = cv2.imread(str(approval_result))
    sample_highlighted = cv2.imread(str(sample_result))

    side_result = highlighter.create_side_by_side(
        approval_highlighted,
        sample_highlighted,
        side_output,
        title="APPROVAL vs SAMPLE - LINE BY LINE QC",
    )

    return {
        "approval": approval_result,
        "sample": sample_result,
        "side_by_side": side_result,
    }


def compare_labels(approval_path, sample_paths):
    ocr = OCREngine()
    detector = FieldDetector()
    comparator = ComparisonEngine()
    logo_checker = LogoChecker()
    barcode_checker = BarcodeChecker()
    highlighter = ImageHighlighter()

    approval_path = Path(approval_path)
    sample_paths = [Path(x) for x in sample_paths]

    if not approval_path.exists():
        raise FileNotFoundError(f"Approval label not found: {approval_path}")

    approval_ocr = ocr.extract(approval_path)
    approval_text = approval_ocr.get("text", "") or ""
    approval_words = approval_ocr.get("words", []) or []
    approval_fields = detector.extract_fields(approval_text)

    samples = []

    for sample_path in sample_paths:
        if not sample_path.exists():
            raise FileNotFoundError(f"Sample label not found: {sample_path}")

        sample_ocr = ocr.extract(sample_path)
        sample_text = sample_ocr.get("text", "") or ""
        sample_words = sample_ocr.get("words", []) or []
        sample_fields = detector.extract_fields(sample_text)

        field_result = comparator.compare_fields(
            approval_fields,
            sample_fields,
        )
        text_result = comparator.compare_text(
            approval_text,
            sample_text,
        )
        comparison = comparator.merge_field_and_text(
            field_result,
            text_result,
        )

        # Add complete line-level audit even when fields were detected.
        comparison["line_result"] = text_result.get("line_result", {})
        comparison["visual_differences"] = text_result.get(
            "visual_differences", []
        )
        line = comparison["line_result"]
        comparison["line_mismatched"] = line.get("mismatched", 0)
        comparison["line_missing"] = line.get("missing", 0)
        comparison["line_extra"] = line.get("extra", 0)

        try:
            logo_result = logo_checker.compare(
                approval_path, sample_path
            )
        except Exception as exc:
            logo_result = {
                "status": "NOT_CHECKED",
                "message": str(exc),
            }

        try:
            barcode_result = barcode_checker.compare(
                approval_path, sample_path
            )
        except Exception as exc:
            barcode_result = {
                "status": "NOT_CHECKED",
                "message": str(exc),
            }

        highlighted = create_highlighted_images(
            highlighter,
            approval_path,
            sample_path,
            approval_words,
            sample_words,
            comparison,
        )

        score = comparator.calculate_overall_score(
            comparison,
            logo_result,
            barcode_result,
        )
        status = calculate_overall_status(
            comparator,
            score,
            comparison,
            logo_result,
            barcode_result,
        )

        samples.append({
            "filename": sample_path.name,
            "image_url": make_browser_url(sample_path),
            "ocr_text": sample_text,
            "ocr_confidence": round(float(sample_ocr.get("confidence", 0) or 0), 1),
            "fields": sample_fields,
            "field_result": comparison,
            "line_result": comparison.get("line_result", {}),
            "visual_differences": comparison.get("visual_differences", []),
            "logo": logo_result,
            "barcode": barcode_result,
            "highlighted_approval_url": make_browser_url(highlighted["approval"]),
            "highlighted_sample_url": make_browser_url(highlighted["sample"]),
            "side_by_side_url": make_browser_url(highlighted["side_by_side"]),
            "score": round(score * 100, 1),
            "status": status,
            "matched": comparison.get("matched", 0),
            "mismatched": comparison.get("mismatched", 0),
            "missing": comparison.get("missing", 0),
            "extra": comparison.get("extra", 0),
            "not_checked": comparison.get("not_checked", 0),
            "total": comparison.get("total", 0),
            "mismatches": comparison.get("mismatches", []),
        })

    return {
        "approval": {
            "filename": approval_path.name,
            "image_url": make_browser_url(approval_path),
            "ocr_text": approval_text,
            "ocr_confidence": round(float(approval_ocr.get("confidence", 0) or 0), 1),
            "fields": approval_fields,
        },
        # Top-level aliases make the template robust.
        "approval_image": make_browser_url(approval_path),
        "approval_path": str(approval_path),
        "samples": samples,
        "samples_count": len(samples),
    }
