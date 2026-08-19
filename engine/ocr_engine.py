"""
=========================================================
LABEL QC CHECKER PRO
OCR ENGINE
=========================================================
High-accuracy OCR extraction for garment/carton labels.
Render-friendly Tesseract handling.
=========================================================
"""

from pathlib import Path
import os
import re
import shutil
import logging

import cv2
import numpy as np
import pytesseract

from config import (
    OCR_LANGUAGE,
    OCR_PSM,
    OCR_MIN_CONFIDENCE,
    OCR_USE_EASYOCR,
    OCR_GPU,
)

logger = logging.getLogger(__name__)


class OCREngine:

    def __init__(self):
        self.easy_reader = None
        self.tesseract_path = self.find_tesseract()

        if self.tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = self.tesseract_path

        self.language = OCR_LANGUAGE or "eng"
        self.psm = OCR_PSM or 6
        self.min_confidence = (
            OCR_MIN_CONFIDENCE
            if OCR_MIN_CONFIDENCE is not None
            else 20
        )
        self.use_easyocr = bool(OCR_USE_EASYOCR)
        self.gpu = bool(OCR_GPU)

        # Render has very limited CPU/time compared with a local machine.
        # On Render we use a much shorter Tesseract timeout and a smaller
        # number of OCR attempts, then fall back to EasyOCR if available.
        self.is_render = bool(
            os.getenv("RENDER")
            or os.getenv("RENDER_SERVICE_ID")
            or os.getenv("RENDER_INSTANCE_ID")
        )

        default_timeout = "3" if self.is_render else "5"
        try:
            self.tesseract_timeout = max(
                1,
                int(os.getenv("TESSERACT_TIMEOUT", default_timeout))
            )
        except (TypeError, ValueError):
            self.tesseract_timeout = 3 if self.is_render else 5

        # A timeout means the Tesseract subprocess was killed.  Do not keep
        # retrying the same expensive operation on Render.
        self._tesseract_timeout_seen = False

    # =====================================================
    # FIND TESSERACT
    # =====================================================

    def find_tesseract(self):
        path = shutil.which("tesseract")
        if path:
            return path

        possible_paths = [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
            os.path.expandvars(
                r"%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe"
            ),
            os.path.expandvars(
                r"%LOCALAPPDATA%\Tesseract-OCR\tesseract.exe"
            ),
            r"C:\Tesseract-OCR\tesseract.exe",
            "/usr/bin/tesseract",
            "/usr/local/bin/tesseract",
        ]

        for path in possible_paths:
            if os.path.isfile(path):
                return path

        logger.warning("Tesseract executable was not found.")
        return None

    # =====================================================
    # CHECK TESSERACT
    # =====================================================

    def is_tesseract_available(self):
        if not self.tesseract_path:
            return False

        try:
            version = pytesseract.get_tesseract_version()
            logger.info("Tesseract detected: %s", version)
            return version is not None
        except Exception as exc:
            logger.warning("Tesseract is not available: %s", exc)
            return False

    # =====================================================
    # LOAD IMAGE
    # =====================================================

    def load_image(self, path):
        if path is None:
            raise ValueError("Image path cannot be None.")

        path = Path(path)

        if not path.exists():
            raise ValueError(f"Image does not exist: {path}")

        if not path.is_file():
            raise ValueError(f"Path is not a file: {path}")

        image = cv2.imread(str(path), cv2.IMREAD_COLOR)

        if image is None or image.size == 0:
            raise ValueError(f"Unable to read image: {path}")

        return image

    # =====================================================
    # UPSCALE
    # =====================================================

    def upscale(self, image):
        if image is None:
            raise ValueError("Image cannot be None.")

        height, width = image.shape[:2]

        if width >= 1800:
            return image

        scale = 2.0

        return cv2.resize(
            image,
            (
                int(width * scale),
                int(height * scale),
            ),
            interpolation=cv2.INTER_CUBIC,
        )

    # =====================================================
    # PREPROCESSING
    # =====================================================

    def preprocess_variants(self, image):
        if image is None:
            raise ValueError("Image cannot be None.")

        variants = []

        enlarged = self.upscale(image)
        variants.append(("original", enlarged))

        gray = cv2.cvtColor(
            enlarged,
            cv2.COLOR_BGR2GRAY,
        )
        variants.append(("gray", gray))

        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8),
        )
        enhanced = clahe.apply(gray)
        variants.append(("clahe", enhanced))

        adaptive = cv2.adaptiveThreshold(
            enhanced,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            11,
        )
        variants.append(("adaptive", adaptive))

        return variants

    def preprocess(self, image):
        variants = self.preprocess_variants(image)

        for name, processed in variants:
            if name == "adaptive":
                return processed

        return variants[0][1]

    # =====================================================
    # CLEAN OCR WORD
    # =====================================================

    def clean_ocr_word(self, text):
        if text is None:
            return ""

        text = str(text).strip()

        if not text:
            return ""

        text = text.replace("\u00a0", " ")
        text = re.sub(r"\s+", " ", text)

        return text.strip()

    # =====================================================
    # TESSERACT WORD-LEVEL OCR
    # =====================================================

    def _extract_tesseract_data(self, image, psm=None):
        if not self.is_tesseract_available():
            return {
                "text": "",
                "words": [],
                "confidence": 0.0,
                "available": False,
                "error": "Tesseract is not available",
            }

        if psm is None:
            psm = self.psm

        try:
            psm = int(psm)
        except (ValueError, TypeError):
            psm = 6

        config = f"--oem 3 --psm {psm}"

        try:
            data = pytesseract.image_to_data(
                image,
                lang=self.language,
                config=config,
                output_type=pytesseract.Output.DICT,
                timeout=self.tesseract_timeout,
            )

        except RuntimeError as exc:
            logger.warning(
                "Tesseract timed out/failed. PSM=%s Error=%s",
                psm,
                exc,
            )
            return {
                "text": "",
                "words": [],
                "confidence": 0.0,
                "available": False,
                "error": str(exc),
            }

        except Exception as exc:
            logger.exception(
                "Tesseract OCR failed. PSM=%s Error=%s",
                psm,
                exc,
            )
            return {
                "text": "",
                "words": [],
                "confidence": 0.0,
                "available": False,
                "error": str(exc),
            }

        if not isinstance(data, dict):
            return {
                "text": "",
                "words": [],
                "confidence": 0.0,
                "available": False,
                "error": "Invalid Tesseract response",
            }

        words = []
        text_items = data.get("text", [])

        if not isinstance(text_items, list):
            text_items = []

        for i in range(len(text_items)):
            text = self.clean_ocr_word(text_items[i])

            if not text:
                continue

            try:
                confidence = float(data.get("conf", [])[i])
            except (ValueError, TypeError, IndexError):
                confidence = 0.0

            if confidence < 0:
                confidence = 0.0

            try:
                x = int(data.get("left", [])[i])
                y = int(data.get("top", [])[i])
                w = int(data.get("width", [])[i])
                h = int(data.get("height", [])[i])
            except (ValueError, TypeError, IndexError):
                x = y = w = h = 0

            try:
                block_num = int(data.get("block_num", [])[i])
            except (ValueError, TypeError, IndexError):
                block_num = 0

            try:
                par_num = int(data.get("par_num", [])[i])
            except (ValueError, TypeError, IndexError):
                par_num = 0

            try:
                line_num = int(data.get("line_num", [])[i])
            except (ValueError, TypeError, IndexError):
                line_num = 0

            words.append({
                "text": text,
                "confidence": round(confidence, 2),
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "x2": x + w,
                "y2": y + h,
                "block_num": block_num,
                "par_num": par_num,
                "line_num": line_num,
            })

        full_text = self.build_line_text(words)

        confidence_values = [
            word["confidence"]
            for word in words
            if word["confidence"] > 0
        ]

        average_confidence = (
            sum(confidence_values) / len(confidence_values)
            if confidence_values
            else 0.0
        )

        return {
            "text": full_text,
            "words": words,
            "confidence": round(average_confidence, 2),
            "available": True,
            "psm": psm,
        }

    # =====================================================
    # SINGLE TESSERACT
    # =====================================================

    def extract_tesseract(self, image):
        return self._extract_tesseract_data(
            image,
            psm=self.psm,
        )

    # =====================================================
    # MULTI-PSM TESSERACT
    # =====================================================

    def extract_tesseract_multi(self, image):
        if not self.is_tesseract_available():
            return {
                "text": "",
                "words": [],
                "confidence": 0.0,
                "available": False,
                "error": "Tesseract is not available",
            }

        try:
            configured_psm = int(self.psm)
        except (ValueError, TypeError):
            configured_psm = 6

        # Local: keep the existing multi-PSM behaviour.
        # Render: one PSM attempt is enough. If it times out, the caller
        # immediately falls back to EasyOCR instead of burning CPU on retries.
        psms = [configured_psm]
        if not self.is_render and 6 not in psms:
            psms.append(6)

        results = []

        for psm in psms:
            try:
                result = self._extract_tesseract_data(image, psm=psm)
            except Exception as exc:
                logger.exception(
                    "Tesseract PSM %s failed: %s",
                    psm,
                    exc,
                )
                continue

            if not isinstance(result, dict):
                continue

            if not result.get("available", False):
                error = str(result.get("error", "")).lower()
                if "timeout" in error or "timed out" in error:
                    self._tesseract_timeout_seen = True
                    # Do not retry another PSM after a timeout on Render.
                    if self.is_render:
                        break
                continue

            if not result.get("text", "").strip():
                continue

            result["psm"] = psm
            results.append(result)

            # Render only needs the first successful OCR result.
            if self.is_render:
                break

        if not results:
            return {
                "text": "",
                "words": [],
                "confidence": 0.0,
                "available": True,
                "error": "Tesseract returned no usable text",
            }

        return max(results, key=self.score_ocr_result)

    # =====================================================
    # BUILD LINE TEXT
    # =====================================================

    def build_line_text(self, words):
        if not words:
            return ""

        groups = {}

        for word in words:
            if not isinstance(word, dict):
                continue

            key = (
                word.get("block_num", 0),
                word.get("par_num", 0),
                word.get("line_num", 0),
            )

            groups.setdefault(key, []).append(word)

        lines = []

        for line_words in groups.values():
            line_words.sort(
                key=lambda item: item.get("x", 0)
            )

            text_parts = [
                str(word.get("text", "")).strip()
                for word in line_words
                if str(word.get("text", "")).strip()
            ]

            line_text = re.sub(
                r"\s+",
                " ",
                " ".join(text_parts),
            ).strip()

            if line_text:
                min_y = min(
                    word.get("y", 0)
                    for word in line_words
                )
                lines.append((min_y, line_text))

        lines.sort(key=lambda item: item[0])

        return "\n".join(
            line[1]
            for line in lines
        )

    # =====================================================
    # SCORE OCR RESULT
    # =====================================================

    def score_ocr_result(self, result):
        if not isinstance(result, dict):
            return 0.0

        text = str(
            result.get("text", "")
        ).strip()

        if not text:
            return 0.0

        try:
            confidence = float(
                result.get("confidence", 0.0) or 0.0
            )
        except (ValueError, TypeError):
            confidence = 0.0

        words = result.get("words", [])

        if not isinstance(words, list):
            words = []

        useful_chars = len(
            re.findall(
                r"[A-Za-z0-9]",
                text,
            )
        )

        text_score = min(
            useful_chars / 20.0,
            10.0,
        )

        word_score = min(
            len(words) / 10.0,
            5.0,
        )

        return confidence + text_score + word_score

    # =====================================================
    # NORMALIZATION
    # =====================================================

    def normalize_ocr_line(self, line):
        if not line:
            return ""

        line = str(line)

        line = (
            line
            .replace("\u00a0", " ")
            .replace("—", "-")
            .replace("–", "-")
            .replace("−", "-")
        )

        return re.sub(
            r"[ \t]+",
            " ",
            line,
        ).strip()

    def normalize_for_duplicate(self, text):
        if not text:
            return ""

        text = str(text).upper()

        return re.sub(
            r"[^A-Z0-9]+",
            "",
            text,
        )

    def is_useful_ocr_line(self, text):
        if not text:
            return False

        return len(
            re.findall(
                r"[A-Za-z0-9]",
                text,
            )
        ) >= 2

    # =====================================================
    # EASY OCR
    # =====================================================

    def extract_easyocr(self, image):
        if not self.use_easyocr:
            return {
                "text": "",
                "words": [],
                "confidence": 0.0,
                "available": False,
            }

        try:
            import easyocr

            if self.easy_reader is None:
                self.easy_reader = easyocr.Reader(
                    [self.language],
                    gpu=self.gpu,
                    verbose=False,
                )

            results = self.easy_reader.readtext(
                image,
                detail=1,
                paragraph=False,
            )

            words = []
            text_parts = []
            confidence_values = []

            for result in results:
                if not isinstance(result, (list, tuple)):
                    continue

                if len(result) < 3:
                    continue

                box, text, confidence = result

                text = str(text).strip()

                try:
                    confidence = float(confidence) * 100
                except (ValueError, TypeError):
                    confidence = 0.0

                if not text:
                    continue

                if confidence < self.min_confidence:
                    continue

                try:
                    xs = [int(point[0]) for point in box]
                    ys = [int(point[1]) for point in box]

                    x1 = min(xs)
                    y1 = min(ys)
                    x2 = max(xs)
                    y2 = max(ys)
                except (ValueError, TypeError, IndexError):
                    continue

                words.append({
                    "text": text,
                    "confidence": round(confidence, 2),
                    "x": x1,
                    "y": y1,
                    "w": x2 - x1,
                    "h": y2 - y1,
                    "x2": x2,
                    "y2": y2,
                })

                text_parts.append(text)
                confidence_values.append(confidence)

            average_confidence = (
                sum(confidence_values) / len(confidence_values)
                if confidence_values
                else 0.0
            )

            return {
                "text": " ".join(text_parts),
                "words": words,
                "confidence": round(
                    average_confidence,
                    2,
                ),
                "available": True,
            }

        except Exception as exc:
            logger.exception(
                "EasyOCR failed: %s",
                exc,
            )

            return {
                "text": "",
                "words": [],
                "confidence": 0.0,
                "available": False,
                "error": str(exc),
            }

    # =====================================================
    # MERGE OCR TEXT
    # =====================================================

    def merge_text(self, tesseract, easy):
        tess_text = (
            str(tesseract.get("text", "")).strip()
            if isinstance(tesseract, dict)
            else ""
        )

        easy_text = (
            str(easy.get("text", "")).strip()
            if isinstance(easy, dict)
            else ""
        )

        if tess_text and not easy_text:
            return tess_text

        if easy_text and not tess_text:
            return easy_text

        if not tess_text and not easy_text:
            return ""

        tess_lines = [
            self.normalize_ocr_line(line)
            for line in tess_text.splitlines()
        ]

        tess_lines = [
            line for line in tess_lines if line
        ]

        easy_lines = [
            self.normalize_ocr_line(line)
            for line in easy_text.splitlines()
        ]

        easy_lines = [
            line for line in easy_lines if line
        ]

        merged = list(tess_lines)

        existing = {
            self.normalize_for_duplicate(line)
            for line in merged
        }

        for line in easy_lines:
            normalized = self.normalize_for_duplicate(line)

            if not normalized:
                continue

            if normalized in existing:
                continue

            if not self.is_useful_ocr_line(line):
                continue

            merged.append(line)
            existing.add(normalized)

        return "\n".join(merged)

    # =====================================================
    # MAIN EXTRACTION
    # =====================================================

    def _map_easy_words(self, easy_words, original_width, original_height, ocr_width, ocr_height):
        """Map EasyOCR boxes from the OCR image back to the original image."""
        if not isinstance(easy_words, list):
            return []

        scale_x = original_width / max(1, ocr_width)
        scale_y = original_height / max(1, ocr_height)

        mapped = []
        for word in easy_words:
            if not isinstance(word, dict):
                continue

            try:
                x = int(round(float(word.get("x", 0)) * scale_x))
                y = int(round(float(word.get("y", 0)) * scale_y))
                w = int(round(float(word.get("w", 0)) * scale_x))
                h = int(round(float(word.get("h", 0)) * scale_y))
            except (TypeError, ValueError):
                continue

            if w <= 0 or h <= 0:
                continue

            x = max(0, min(x, original_width - 1))
            y = max(0, min(y, original_height - 1))
            x2 = min(original_width, x + w)
            y2 = min(original_height, y + h)

            item = dict(word)
            item.update({
                "x": x,
                "y": y,
                "w": max(1, x2 - x),
                "h": max(1, y2 - y),
                "x2": x2,
                "y2": y2,
                "original_x": x,
                "original_y": y,
                "original_w": max(1, x2 - x),
                "original_h": max(1, y2 - y),
                "coordinate_space": "original",
                "scale_x": scale_x,
                "scale_y": scale_y,
            })
            mapped.append(item)

        # EasyOCR does not provide Tesseract's block/paragraph/line numbers.
        # Create physical lines from the Y coordinate so the rest of the
        # comparison engine receives the same line_index/word_index fields.
        mapped.sort(key=lambda item: (item.get("y", 0), item.get("x", 0)))

        line_groups = []
        for item in mapped:
            cy = item.get("y", 0) + item.get("h", 0) / 2
            placed = False

            for group in line_groups:
                avg_cy = group["sum_y"] / len(group["items"])
                avg_h = group["sum_h"] / len(group["items"])
                tolerance = max(8.0, avg_h * 0.65)

                if abs(cy - avg_cy) <= tolerance:
                    group["items"].append(item)
                    group["sum_y"] += cy
                    group["sum_h"] += item.get("h", 0)
                    placed = True
                    break

            if not placed:
                line_groups.append({
                    "items": [item],
                    "sum_y": cy,
                    "sum_h": item.get("h", 0),
                })

        line_groups.sort(
            key=lambda group: min(
                item.get("y", 0) for item in group["items"]
            )
        )

        final_words = []
        for line_index, group in enumerate(line_groups):
            group["items"].sort(key=lambda item: item.get("x", 0))
            for word_index, item in enumerate(group["items"]):
                item["line_index"] = line_index
                item["word_index"] = word_index
                final_words.append(item)

        return final_words

    def extract(self, path):
        """
        Complete OCR pipeline.

        Tesseract remains the primary engine. If Tesseract times out or
        returns no usable words (the Render failure seen in production),
        EasyOCR is used as a fallback. Returned coordinates are always mapped
        back to the original image dimensions.
        """
        image = self.load_image(path)

        original_height, original_width = image.shape[:2]

        variants = self.preprocess_variants(image)
        tesseract_results = []

        # Local keeps the high-accuracy multi-variant pipeline.
        # Render intentionally uses only the original/upscaled image first.
        # This prevents 4 variants x multiple PSM calls from exhausting the
        # Render worker when Tesseract is slow.
        if self.is_render:
            variants_to_try = [
                (name, processed)
                for name, processed in variants
                if name == "original"
            ]
        else:
            variants_to_try = variants

        for variant_name, processed in variants_to_try:
            try:
                result = self.extract_tesseract_multi(processed)
            except Exception as exc:
                logger.exception(
                    "OCR variant %s failed: %s",
                    variant_name,
                    exc,
                )
                continue

            if not isinstance(result, dict):
                continue

            text = str(result.get("text", "")).strip()

            if not result.get("available", False):
                continue

            if not text or not result.get("words"):
                # A successful process with zero words is not useful for QC.
                continue

            result["variant"] = variant_name
            result["_ocr_width"] = processed.shape[1]
            result["_ocr_height"] = processed.shape[0]
            tesseract_results.append(result)

            # Render: stop after the first usable result.
            if self.is_render:
                break

        if tesseract_results:
            best_tesseract = max(
                tesseract_results,
                key=self.score_ocr_result,
            )
        else:
            best_tesseract = {
                "text": "",
                "words": [],
                "confidence": 0.0,
                "available": self.is_tesseract_available(),
            }

        tesseract_words = best_tesseract.get("words", [])
        if not isinstance(tesseract_words, list):
            tesseract_words = []

        # -----------------------------------------------------
        # EASY OCR FALLBACK
        # -----------------------------------------------------
        # Respect OCR_USE_EASYOCR locally. On Render, however, automatically
        # enable EasyOCR as the fallback when Tesseract produced no words.
        # This is the key fix for the production "words=0 / lines=0" result.
        need_easy_fallback = len(tesseract_words) == 0

        easy_result = {
            "text": "",
            "words": [],
            "confidence": 0.0,
            "available": False,
        }

        if self.use_easyocr or (self.is_render and need_easy_fallback):
            try:
                easy_image = self.upscale(image)
                easy_result = self.extract_easyocr(easy_image)
                easy_result["_ocr_width"] = easy_image.shape[1]
                easy_result["_ocr_height"] = easy_image.shape[0]
            except Exception as exc:
                logger.exception("EasyOCR fallback failed: %s", exc)
                easy_result = {
                    "text": "",
                    "words": [],
                    "confidence": 0.0,
                    "available": False,
                    "error": str(exc),
                }

        # -----------------------------------------------------
        # SELECT PRIMARY OCR RESULT
        # -----------------------------------------------------
        if tesseract_words:
            primary_words = tesseract_words
            ocr_width = int(
                best_tesseract.get("_ocr_width", original_width)
                or original_width
            )
            ocr_height = int(
                best_tesseract.get("_ocr_height", original_height)
                or original_height
            )
            primary_confidence = float(
                best_tesseract.get("confidence", 0.0) or 0.0
            )
            primary_engine = "tesseract"
        else:
            easy_words = easy_result.get("words", [])
            if not isinstance(easy_words, list):
                easy_words = []

            primary_words = easy_words
            ocr_width = int(
                easy_result.get("_ocr_width", original_width)
                or original_width
            )
            ocr_height = int(
                easy_result.get("_ocr_height", original_height)
                or original_height
            )
            primary_confidence = float(
                easy_result.get("confidence", 0.0) or 0.0
            )
            primary_engine = "easyocr"

        # -----------------------------------------------------
        # MAP WORDS TO ORIGINAL IMAGE
        # -----------------------------------------------------
        if primary_engine == "easyocr":
            mapped_words = self._map_easy_words(
                primary_words,
                original_width,
                original_height,
                ocr_width,
                ocr_height,
            )
        else:
            scale_x = original_width / max(1, ocr_width)
            scale_y = original_height / max(1, ocr_height)

            mapped_words = []

            for word in primary_words:
                if not isinstance(word, dict):
                    continue

                try:
                    x = int(round(float(word.get("x", 0)) * scale_x))
                    y = int(round(float(word.get("y", 0)) * scale_y))
                    width = int(round(float(word.get("w", 0)) * scale_x))
                    height = int(round(float(word.get("h", 0)) * scale_y))
                except (TypeError, ValueError):
                    continue

                if width <= 0 or height <= 0:
                    continue

                x = max(0, min(x, original_width - 1))
                y = max(0, min(y, original_height - 1))
                x2 = min(original_width, x + width)
                y2 = min(original_height, y + height)

                item = dict(word)
                item.update({
                    "x": x,
                    "y": y,
                    "w": max(1, x2 - x),
                    "h": max(1, y2 - y),
                    "x2": x2,
                    "y2": y2,
                    "original_x": x,
                    "original_y": y,
                    "original_w": max(1, x2 - x),
                    "original_h": max(1, y2 - y),
                    "coordinate_space": "original",
                    "scale_x": scale_x,
                    "scale_y": scale_y,
                })
                mapped_words.append(item)

            # Group Tesseract words into physical lines.
            groups = {}
            for word in mapped_words:
                key = (
                    word.get("block_num", 0),
                    word.get("par_num", 0),
                    word.get("line_num", 0),
                )
                groups.setdefault(key, []).append(word)

            ordered_groups = sorted(
                groups.values(),
                key=lambda group: (
                    min(item.get("y", 0) for item in group),
                    min(item.get("x", 0) for item in group),
                ),
            )

            for line_index, line_words in enumerate(ordered_groups):
                line_words.sort(key=lambda item: item.get("x", 0))
                for word_index, item in enumerate(line_words):
                    item["line_index"] = line_index
                    item["word_index"] = word_index

            mapped_words.sort(
                key=lambda item: (
                    item.get("line_index", 0),
                    item.get("x", 0),
                )
            )

        # -----------------------------------------------------
        # FINAL TEXT / LINES
        # -----------------------------------------------------
        final_text = self.build_line_text(mapped_words)

        # If EasyOCR is primary, build_line_text works from the generated
        # line_index values above. For Tesseract it preserves the original
        # physical line grouping.
        lines = []
        line_groups_final = {}

        for word in mapped_words:
            line_index = int(word.get("line_index", 0))
            line_groups_final.setdefault(line_index, []).append(word)

        for line_index in sorted(line_groups_final):
            group = sorted(
                line_groups_final[line_index],
                key=lambda item: item.get("x", 0),
            )
            line_text = " ".join(
                str(word.get("text", "")).strip()
                for word in group
                if str(word.get("text", "")).strip()
            ).strip()

            if line_text:
                lines.append({
                    "line_index": line_index,
                    "text": line_text,
                    "words": group,
                })

        tess_confidence = float(
            best_tesseract.get("confidence", 0.0) or 0.0
        )
        easy_confidence = float(
            easy_result.get("confidence", 0.0) or 0.0
        )

        confidence_values = [
            value for value in (tess_confidence, easy_confidence)
            if value > 0
        ]

        final_confidence = (
            sum(confidence_values) / len(confidence_values)
            if confidence_values
            else primary_confidence
        )

        logger.info(
            "OCR completed: engine=%s words=%d lines=%d confidence=%.1f",
            primary_engine,
            len(mapped_words),
            len(lines),
            final_confidence,
        )

        return {
            "text": final_text.strip(),
            "confidence": round(final_confidence, 1),
            "words": mapped_words,
            "lines": lines,
            "coordinate_space": "original",
            "image_width": original_width,
            "image_height": original_height,
            "engine": primary_engine,
            "tesseract": best_tesseract,
            "easyocr": easy_result,
        }
