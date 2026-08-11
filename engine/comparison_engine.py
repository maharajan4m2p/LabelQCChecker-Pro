"""
Label QC Checker Pro - Line-by-Line Comparison Engine
"""

import re
from difflib import SequenceMatcher
from config import FIELD_MATCH_THRESHOLD, OVERALL_PASS_THRESHOLD


class ComparisonEngine:
    """Strict but OCR-tolerant Approval vs Sample comparison."""

    def __init__(self):
        self.field_match_threshold = float(FIELD_MATCH_THRESHOLD)
        self.overall_pass_threshold = float(OVERALL_PASS_THRESHOLD)
        self.ocr_replacements = {
            "—": "-", "–": "-", "−": "-",
            "“": '"', "”": '"', "‘": "'", "’": "'",
            "|": "I",
        }

    def normalize_text(self, value):
        if value is None:
            return ""
        value = str(value).upper()
        for old, new in self.ocr_replacements.items():
            value = value.replace(old, new)
        value = re.sub(r"\s+", " ", value)
        return value.strip()

    def normalize(self, value):
        return re.sub(r"[^A-Z0-9]", "", self.normalize_text(value))

    def normalize_for_comparison(self, value):
        return self.normalize(value)

    def tokenize(self, value):
        value = self.normalize_text(value)
        return re.findall(r"[A-Z0-9]+(?:[-./%][A-Z0-9]+)*", value) if value else []

    def extract_numbers(self, value):
        return re.findall(r"\d+(?:\.\d+)?", str(value or ""))

    def has_numeric_difference(self, approval, sample):
        a, s = self.extract_numbers(approval), self.extract_numbers(sample)
        if not a:
            return False
        if not s:
            return True
        return a != s

    def similarity(self, left, right):
        a, b = self.normalize(left), self.normalize(right)
        if not a and not b:
            return 1.0
        if not a or not b:
            return 0.0
        return SequenceMatcher(None, a, b, autojunk=False).ratio()

    def tolerant_equal(self, approval, sample):
        a, b = self.normalize(approval), self.normalize(sample)
        return bool(a and b and a == b)

    def get_character_differences(self, approval, sample):
        a, b = self.normalize_text(approval), self.normalize_text(sample)
        out = []
        for tag, i1, i2, j1, j2 in SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
            if tag != "equal":
                out.append({
                    "type": tag, "approval": a[i1:i2], "sample": b[j1:j2],
                    "approval_start": i1, "approval_end": i2,
                    "sample_start": j1, "sample_end": j2,
                })
        return out

    def get_word_differences(self, approval, sample):
        aw, sw = self.tokenize(approval), self.tokenize(sample)
        out = []
        for tag, i1, i2, j1, j2 in SequenceMatcher(
            None, aw, sw, autojunk=False
        ).get_opcodes():
            if tag == "equal":
                continue
            out.append({
                "type": tag,
                "approval": aw[i1:i2],
                "sample": sw[j1:j2],
                "approval_text": " ".join(aw[i1:i2]),
                "sample_text": " ".join(sw[j1:j2]),
            })
        return out

    def compare_text_values(self, approval, sample):
        a, s = str(approval or "").strip(), str(sample or "").strip()
        if not a and not s:
            return self._empty("NOT_CHECKED", None)
        if a and not s:
            return self._empty("MISSING", 0.0)
        if not a and s:
            return self._empty("EXTRA", 0.0)

        numeric = self.has_numeric_difference(a, s)
        score = self.similarity(a, s)
        words = self.get_word_differences(a, s)
        chars = self.get_character_differences(a, s)

        status = "MATCH" if self.tolerant_equal(a, s) else (
            "MISMATCH" if numeric or score < 0.98 else "MATCH"
        )
        return {
            "status": status, "score": round(score * 100, 2),
            "differences": words, "word_differences": words,
            "character_differences": chars,
            "numeric_difference": numeric,
        }

    @staticmethod
    def _empty(status, score):
        return {
            "status": status, "score": score, "differences": [],
            "word_differences": [], "character_differences": [],
            "numeric_difference": False,
        }

    def compare_value(self, approval_value, sample_value):
        result = self.compare_text_values(approval_value, sample_value)
        if result["status"] == "MATCH" and result["score"] is not None:
            if result.get("numeric_difference"):
                result["status"] = "MISMATCH"
            elif result["score"] < self.field_match_threshold * 100:
                result["status"] = "MISMATCH"
        return result

    def compare_fields(self, approval_fields, sample_fields):
        approval_fields, sample_fields = approval_fields or {}, sample_fields or {}
        rows, mismatches = [], []
        counts = {"matched": 0, "mismatched": 0, "missing": 0, "extra": 0, "not_checked": 0}

        fields = list(dict.fromkeys(list(approval_fields) + list(sample_fields)))
        for field in fields:
            a = str(approval_fields.get(field, "") or "").strip()
            s = str(sample_fields.get(field, "") or "").strip()
            c = self.compare_value(a, s)
            status = c["status"]
            if status == "MATCH":
                counts["matched"] += 1

            elif status == "MISMATCH":
                counts["mismatched"] += 1

            elif status == "MISSING":
                counts["missing"] += 1

            elif status == "EXTRA":
                counts["extra"] += 1

            elif status == "NOT_CHECKED":
                counts["not_checked"] += 1

            row = {
                "field": field, "approval": a, "sample": s, "status": status,
                "score": c.get("score"), "differences": c.get("differences", []),
                "word_differences": c.get("word_differences", []),
                "character_differences": c.get("character_differences", []),
            }
            if status != "NOT_CHECKED":
                rows.append(row)
            if status in {"MISMATCH", "MISSING", "EXTRA"}:
                mismatches.append(row.copy())

        comparable = counts["matched"] + counts["mismatched"] + counts["missing"] + counts["extra"]
        return {
            "rows": rows, **counts, "total": comparable,
            "field_score": round(counts["matched"] / comparable, 4) if comparable else 0.0,
            "mismatches": mismatches,
        }

    def _line_token_differences(self, a, s, ai, si):
        aw, sw = self.tokenize(a), self.tokenize(s)
        result = []
        for tag, i1, i2, j1, j2 in SequenceMatcher(None, aw, sw, autojunk=False).get_opcodes():
            if tag == "equal":
                continue
            result.append({
                "type": tag,
                "approval": aw[i1:i2],
                "sample": sw[j1:j2],
                "approval_text": " ".join(aw[i1:i2]),
                "sample_text": " ".join(sw[j1:j2]),
                "approval_line": ai,
                "sample_line": si,
            })
        return result

    def compare_lines(self, approval_text, sample_text):
        """
        Compare physical OCR reading-order lines.
        Returns line numbers and exact changed/missing/extra tokens.
        """
        alines = [self.normalize_text(x) for x in str(approval_text or "").splitlines() if self.normalize_text(x)]
        slines = [self.normalize_text(x) for x in str(sample_text or "").splitlines() if self.normalize_text(x)]

        ops = SequenceMatcher(None, alines, slines, autojunk=False).get_opcodes()
        line_rows, visual = [], []
        counts = {"matched": 0, "mismatched": 0, "missing": 0, "extra": 0}

        for tag, i1, i2, j1, j2 in ops:
            if tag == "equal":
                for ai, sj in zip(range(i1, i2), range(j1, j2)):
                    counts["matched"] += 1
                    line_rows.append({
                        "line": ai + 1, "approval_line": ai, "sample_line": sj,
                        "approval": alines[ai], "sample": slines[sj],
                        "status": "MATCH", "differences": [],
                    })
                continue

            if tag == "replace":
                n = max(i2 - i1, j2 - j1)
                for k in range(n):
                    ai = i1 + k if i1 + k < i2 else None
                    sj = j1 + k if j1 + k < j2 else None
                    a = alines[ai] if ai is not None else ""
                    s = slines[sj] if sj is not None else ""
                    if ai is not None and sj is not None:
                        status = "MATCH" if self.tolerant_equal(a, s) else "MISMATCH"
                        if status == "MATCH":
                            counts["matched"] += 1
                        else:
                            counts["mismatched"] += 1
                            visual.extend(self._line_token_differences(a, s, ai, sj))
                    elif ai is not None:
                        status = "MISSING"; counts["missing"] += 1
                        toks = self.tokenize(a)
                        visual.append({
                            "type": "delete", "approval": toks, "sample": [],
                            "approval_text": " ".join(toks), "sample_text": "",
                            "approval_line": ai, "sample_line": None,
                        })
                    else:
                        status = "EXTRA"; counts["extra"] += 1
                        toks = self.tokenize(s)
                        visual.append({
                            "type": "insert", "approval": [], "sample": toks,
                            "approval_text": "", "sample_text": " ".join(toks),
                            "approval_line": None, "sample_line": sj,
                        })
                    line_rows.append({
                        "line": (
                            ai + 1
                            if isinstance(ai, int)
                            else(
                                sj+1
                                if isinstance(sj, int)
                                else 0)),
                        "approval_line": ai, 
                        "sample_line": sj,
                        "approval": a,
                        "sample": s, 
                        "status": status,
                        "differences": self._line_token_differences(a, s, ai, sj)
                        if ai is not None and sj is not None else [],
                    })
                continue

            if tag == "delete":
                for ai in range(i1, i2):
                    counts["missing"] += 1
                    toks = self.tokenize(alines[ai])
                    visual.append({
                        "type": "delete", "approval": toks, "sample": [],
                        "approval_text": " ".join(toks), "sample_text": "",
                        "approval_line": ai, "sample_line": None,
                    })
                    line_rows.append({
                        "line": ai + 1, "approval_line": ai, "sample_line": None,
                        "approval": alines[ai], "sample": "", "status": "MISSING",
                        "differences": [],
                    })
                continue

            if tag == "insert":
                for sj in range(j1, j2):
                    counts["extra"] += 1
                    toks = self.tokenize(slines[sj])
                    visual.append({
                        "type": "insert", "approval": [], "sample": toks,
                        "approval_text": "", "sample_text": " ".join(toks),
                        "approval_line": None, "sample_line": sj,
                    })
                    line_rows.append({
                        "line": sj + 1, "approval_line": None, "sample_line": sj,
                        "approval": "", "sample": slines[sj], "status": "EXTRA",
                        "differences": [],
                    })

        total = sum(counts.values())
        return {
            "lines": line_rows,
            **counts,
            "total": total,
            "score": round(counts["matched"] / total, 4) if total else 0.0,
            "visual_differences": visual,
        }

    def compare_text(self, approval_text, sample_text):
        line_result = self.compare_lines(approval_text, sample_text)
        status = "MATCH" if line_result["mismatched"] == 0 and line_result["missing"] == 0 and line_result["extra"] == 0 else "MISMATCH"
        return {
            "rows": line_result["lines"],
            "matched": line_result["matched"],
            "mismatched": line_result["mismatched"],
            "missing": line_result["missing"],
            "extra": line_result["extra"],
            "not_checked": 0,
            "total": line_result["total"],
            "field_score": line_result["score"],
            "mismatches": [
                {
                    "field": f"LINE {x.get('line')}",
                    "approval": x.get("approval", ""),
                    "sample": x.get("sample", ""),
                    "status": x.get("status"),
                    "score": None,
                    "word_differences": x.get("differences", []),
                    "differences": x.get("differences", []),
                }
                for x in line_result["lines"]
                if x.get("status") in {"MISMATCH", "MISSING", "EXTRA"}
            ],
            "line_result": line_result,
            "visual_differences": line_result["visual_differences"],
        }

    def merge_field_and_text(self, field_result, text_result):
        # Keep field results for business fields, but preserve line-level
        # results for visual highlighting and auditing.
        result = dict(field_result or {})
        if not result.get("total", 0):
            result = dict(text_result or {})
        else:
            result["line_result"] = text_result.get("line_result", {})
            result["visual_differences"] = text_result.get("visual_differences", [])
            # Any definite line difference is a real QC difference.
            line = result["line_result"]
            result["line_mismatched"] = line.get("mismatched", 0)
            result["line_missing"] = line.get("missing", 0)
            result["line_extra"] = line.get("extra", 0)
        return result

    def calculate_overall_score(self, comparison_result, logo_result=None, barcode_result=None):
        scores, weights = [], []
        if comparison_result and comparison_result.get("total", 0) > 0:
            scores.append(float(comparison_result.get("field_score", 0.0)))
            weights.append(0.85)
        for result, weight in ((logo_result, .10), (barcode_result, .05)):
            status = str((result or {}).get("status", "")).upper()
            if status in {"PASS", "FAIL"}:
                scores.append(1.0 if status == "PASS" else 0.0); weights.append(weight)
        if not scores:
            return 0.0
        return max(0.0, min(1.0, sum(s*w for s,w in zip(scores,weights))/sum(weights)))

    def is_overall_pass(self, score, comparison_result=None, logo_result=None, barcode_result=None):
        c = comparison_result or {}
        if any(c.get(k, 0) > 0 for k in ("mismatched","missing","extra","line_mismatched","line_missing","line_extra")):
            return False
        if str((logo_result or {}).get("status","")).upper() == "FAIL": return False
        if str((barcode_result or {}).get("status","")).upper() == "FAIL": return False
        return float(score) >= self.overall_pass_threshold

    def build_summary(self, comparison_result, logo_result=None, barcode_result=None):
        comparison_result = comparison_result or {}
        score = self.calculate_overall_score(comparison_result, logo_result, barcode_result)
        return {
            "score": round(score*100,1),
            "score_fraction": round(score,4),
            "status": "PASS" if self.is_overall_pass(score, comparison_result, logo_result, barcode_result) else "FAIL",
            "matched": comparison_result.get("matched",0),
            "mismatched": comparison_result.get("mismatched",0),
            "missing": comparison_result.get("missing",0),
            "extra": comparison_result.get("extra",0),
            "not_checked": comparison_result.get("not_checked",0),
            "total": comparison_result.get("total",0),
        }
