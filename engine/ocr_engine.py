"""
=========================================================
LABEL QC CHECKER PRO
OCR ENGINE
=========================================================

Purpose:
    High-accuracy OCR extraction for garment/carton labels.

Features:
    - Automatic Tesseract detection
    - Windows Tesseract support
    - Multiple image preprocessing methods
    - Original image OCR
    - Grayscale OCR
    - Adaptive threshold OCR
    - Otsu threshold OCR
    - Upscaling for small text
    - CLAHE enhancement
    - Tesseract word confidence
    - Preserves OCR line structure
    - Optional EasyOCR
    - Multiple OCR result merging
    - Duplicate removal
    - OCR noise filtering
    - Bounding-box information
    - Safe fallback when OCR engine unavailable

=========================================================
"""

from pathlib import Path
import os
import re
import shutil

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


class OCREngine:
    """
    OCR service for label images.

    Main public method:

        extract(path)

    Returns:

        {
            "text": "...",
            "confidence": 95.2
        }
    """

    # =====================================================
    # INITIALIZATION
    # =====================================================

    def __init__(self):

        self.easy_reader = None

        # Find Tesseract automatically.
        self.tesseract_path = self.find_tesseract()

        if self.tesseract_path:

            pytesseract.pytesseract.tesseract_cmd = (
                self.tesseract_path
            )

        # -------------------------------------------------
        # Configuration
        # -------------------------------------------------

        self.language = (
            OCR_LANGUAGE
            if OCR_LANGUAGE
            else "eng"
        )

        self.psm = (
            OCR_PSM
            if OCR_PSM
            else 6
        )

        self.min_confidence = (
            OCR_MIN_CONFIDENCE
            if OCR_MIN_CONFIDENCE is not None
            else 20
        )

        self.use_easyocr = bool(
            OCR_USE_EASYOCR
        )

        self.gpu = bool(
            OCR_GPU
        )

    # =====================================================
    # FIND TESSERACT
    # =====================================================

    def find_tesseract(self):
        """
        Find Tesseract OCR automatically on Windows.

        Checks:
            1. PATH
            2. Standard Program Files
            3. LocalAppData
        """

        # -------------------------------------------------
        # 1. PATH
        # -------------------------------------------------

        path = shutil.which(
            "tesseract"
        )

        if path:
            return path

        # -------------------------------------------------
        # 2. Common Windows locations
        # -------------------------------------------------

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
        ]

        for path in possible_paths:

            if os.path.isfile(path):
                return path

        return None

    # =====================================================
    # CHECK TESSERACT
    # =====================================================

    def is_tesseract_available(self):
        """
        Check whether Tesseract is actually available.
        """

        if not self.tesseract_path:
            return False

        try:

            version = (
                pytesseract.get_tesseract_version()
            )

            return version is not None

        except Exception:
            return False

    # =====================================================
    # LOAD IMAGE
    # =====================================================

    def load_image(self, path):
        """
        Load an image safely using OpenCV.
        """

        path = Path(path)

        if not path.exists():
            raise ValueError(
                f"Image does not exist: {path}"
            )

        image = cv2.imread(
            str(path)
        )

        if image is None:
            raise ValueError(
                f"Unable to read image: {path}"
            )

        return image

    # =====================================================
    # UPSCALE
    # =====================================================

    def upscale(self, image):
        """
        Upscale small label images.

        This is important for small printed text.
        """

        height, width = image.shape[:2]

        # Do not unnecessarily enlarge already large images.
        if width >= 1800:
            return image

        scale = 2.0

        new_width = int(
            width * scale
        )

        new_height = int(
            height * scale
        )

        return cv2.resize(
            image,
            (new_width, new_height),
            interpolation=cv2.INTER_CUBIC,
        )

    # =====================================================
    # PREPROCESSING
    # =====================================================

    def preprocess_variants(self, image):
        """
        Generate multiple OCR-friendly versions.

        We don't depend on a single thresholding method
        because garment labels can have different:
            - backgrounds
            - lighting
            - fonts
            - print quality
            - borders
            - compression
        """

        variants = []

        # -------------------------------------------------
        # Variant 1
        # Original/upscaled
        # -------------------------------------------------

        enlarged = self.upscale(
            image
        )

        variants.append(
            (
                "original",
                enlarged
            )
        )

        # -------------------------------------------------
        # Grayscale
        # -------------------------------------------------

        gray = cv2.cvtColor(
            enlarged,
            cv2.COLOR_BGR2GRAY,
        )

        variants.append(
            (
                "gray",
                gray
            )
        )

        # -------------------------------------------------
        # CLAHE
        # -------------------------------------------------

        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8),
        )

        enhanced = clahe.apply(
            gray
        )

        variants.append(
            (
                "clahe",
                enhanced
            )
        )

        # -------------------------------------------------
        # Gaussian blur
        # -------------------------------------------------

        blurred = cv2.GaussianBlur(
            enhanced,
            (3, 3),
            0,
        )

        variants.append(
            (
                "blur",
                blurred
            )
        )

        # -------------------------------------------------
        # OTSU
        # -------------------------------------------------

        _, otsu = cv2.threshold(
            enhanced,
            0,
            255,
            cv2.THRESH_BINARY
            + cv2.THRESH_OTSU,
        )

        variants.append(
            (
                "otsu",
                otsu
            )
        )

        # -------------------------------------------------
        # Adaptive threshold
        # -------------------------------------------------

        adaptive = cv2.adaptiveThreshold(
            enhanced,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            11,
        )

        variants.append(
            (
                "adaptive",
                adaptive
            )
        )

        # -------------------------------------------------
        # Mild adaptive threshold
        # -------------------------------------------------

        adaptive_mild = cv2.adaptiveThreshold(
            enhanced,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            21,
            7,
        )

        variants.append(
            (
                "adaptive_mild",
                adaptive_mild
            )
        )

        return variants

    # =====================================================
    # PREPROCESS - BACKWARD COMPATIBILITY
    # =====================================================

    def preprocess(self, image):
        """
        Backward-compatible preprocessing method.

        Returns the primary adaptive-threshold image.
        """

        variants = self.preprocess_variants(
            image
        )

        for name, processed in variants:

            if name == "adaptive":
                return processed

        return variants[0][1]

    # =====================================================
    # CLEAN OCR WORD
    # =====================================================

    def clean_ocr_word(self, text):
        """
        Clean a single OCR word without destroying
        meaningful label characters.
        """

        if text is None:
            return ""

        text = str(text).strip()

        if not text:
            return ""

        # Replace strange Unicode spaces.
        text = text.replace(
            "\u00a0",
            " "
        )

        # Collapse whitespace.
        text = re.sub(
            r"\s+",
            " ",
            text
        )

        return text.strip()

    # =====================================================
    # OCR DATA
    # =====================================================

    def _extract_tesseract_data(
        self,
        image,
        psm=None,
    ):
        """
        Execute Tesseract image_to_data.

        Returns word-level information with:
            text
            confidence
            coordinates
            block
            paragraph
            line
        """

        if not self.is_tesseract_available():

            return {
                "text": "",
                "words": [],
                "confidence": 0,
                "available": False,
            }

        try:

            if psm is None:
                psm = self.psm

            config = (
                f"--oem 3 --psm {psm}"
            )

            data = (
                pytesseract.image_to_data(
                    image,
                    lang=self.language,
                    config=config,
                    output_type=(
                        pytesseract.Output.DICT
                    ),
                )
            )

            words = []

            total_items = len(
                data.get("text", [])
            )

            for i in range(
                total_items
            ):

                raw_text = data[
                    "text"
                ][i]

                text = self.clean_ocr_word(
                    raw_text
                )

                if not text:
                    continue

                # -----------------------------------------
                # Confidence
                # -----------------------------------------

                try:

                    confidence = float(
                        data["conf"][i]
                    )

                except (
                    ValueError,
                    TypeError,
                ):

                    confidence = 0.0

                # Keep low confidence words internally.
                # They can be useful when comparing labels.
                if confidence < 0:
                    confidence = 0.0

                # -----------------------------------------
                # Coordinates
                # -----------------------------------------

                try:

                    x = int(
                        data["left"][i]
                    )

                    y = int(
                        data["top"][i]
                    )

                    w = int(
                        data["width"][i]
                    )

                    h = int(
                        data["height"][i]
                    )

                except (
                    ValueError,
                    TypeError,
                ):

                    x = 0
                    y = 0
                    w = 0
                    h = 0

                # -----------------------------------------
                # Hierarchy
                # -----------------------------------------

                try:
                    block_num = int(
                        data["block_num"][i]
                    )
                except Exception:
                    block_num = 0

                try:
                    par_num = int(
                        data["par_num"][i]
                    )
                except Exception:
                    par_num = 0

                try:
                    line_num = int(
                        data["line_num"][i]
                    )
                except Exception:
                    line_num = 0

                words.append(
                    {
                        "text": text,
                        "confidence": round(
                            confidence,
                            2,
                        ),
                        "x": x,
                        "y": y,
                        "w": w,
                        "h": h,
                        "x2": x + w,
                        "y2": y + h,
                        "block_num": block_num,
                        "par_num": par_num,
                        "line_num": line_num,
                    }
                )

            # ---------------------------------------------
            # Construct line-preserving OCR text.
            # ---------------------------------------------

            full_text = (
                self.build_line_text(
                    words
                )
            )

            confidence_values = [
                word["confidence"]
                for word in words
                if word["confidence"] > 0
            ]

            average_confidence = (
                sum(confidence_values)
                / len(confidence_values)
                if confidence_values
                else 0.0
            )

            return {
                "text": full_text,
                "words": words,
                "confidence": round(
                    average_confidence,
                    2,
                ),
                "available": True,
            }

        except Exception as exc:

            return {
                "text": "",
                "words": [],
                "confidence": 0,
                "available": False,
                "error": str(exc),
            }

    # =====================================================
    # TESSERACT EXTRACTION
    # =====================================================

    def extract_tesseract(
        self,
        image,
    ):
        """
        Run Tesseract OCR using the configured PSM.
        """

        return self._extract_tesseract_data(
            image,
            psm=self.psm,
        )

    # =====================================================
    # MULTI-PSM TESSERACT
    # =====================================================

    def extract_tesseract_multi(
        self,
        image,
    ):
        """
        Run Tesseract with multiple PSM modes.

        PSM:
            6  = uniform block
            11 = sparse text
            12 = sparse text + OSD
            3  = automatic page segmentation

        Different label layouts benefit from different
        segmentation modes.
        """

        if not self.is_tesseract_available():

            return {
                "text": "",
                "words": [],
                "confidence": 0,
                "available": False,
            }

        psms = []

        # Configured PSM first.
        try:
            configured_psm = int(
                self.psm
            )
        except Exception:
            configured_psm = 6

        psms.append(
            configured_psm
        )

        # Additional modes.
        for psm in [6, 11, 12, 3]:

            if psm not in psms:
                psms.append(psm)

        results = []

        for psm in psms:

            result = (
                self._extract_tesseract_data(
                    image,
                    psm=psm,
                )
            )

            if result.get(
                "available",
                False
            ) and result.get(
                "text",
                ""
            ).strip():

                results.append(
                    result
                )

        if not results:

            return {
                "text": "",
                "words": [],
                "confidence": 0,
                "available": True,
            }

        # Choose the best structured result.
        best = max(
            results,
            key=lambda item: (
                self.score_ocr_result(
                    item
                )
            ),
        )

        return best

    # =====================================================
    # BUILD LINE TEXT
    # =====================================================

    def build_line_text(
        self,
        words,
    ):
        """
        Reconstruct OCR text while preserving line breaks.

        This is VERY important for FieldDetector.

        Previous code:

            " ".join(words)

        destroyed line boundaries.

        This version reconstructs:

            BUYER: ABC FASHION LTD.
            VENDOR: VEN-2401
            PO NO: PO/AF/24-25/1001
            STYLE NO: AV-TS-2401
        """

        if not words:
            return ""

        groups = {}

        for word in words:

            key = (
                word.get("block_num", 0),
                word.get("par_num", 0),
                word.get("line_num", 0),
            )

            groups.setdefault(
                key,
                []
            ).append(word)

        lines = []

        for key, line_words in groups.items():

            # Sort left-to-right.
            line_words.sort(
                key=lambda item: (
                    item.get("x", 0)
                )
            )

            line_text = " ".join(
                item["text"]
                for item in line_words
                if item.get("text")
            )

            line_text = re.sub(
                r"\s+",
                " ",
                line_text
            ).strip()

            if line_text:
                lines.append(
                    (
                        min(
                            item.get(
                                "y",
                                0
                            )
                            for item in line_words
                        ),
                        line_text,
                    )
                )

        # Sort lines vertically.
        lines.sort(
            key=lambda item: item[0]
        )

        return "\n".join(
            line[1]
            for line in lines
        )

    # =====================================================
    # EASY OCR INITIALIZATION
    # ==========================
    def build_easyocr_lines(
        self,
        words,
    ):
        """
        Reconstruct EasyOCR results into readable lines.
        """

        if not words:
            return ""

        lines = []

        # Average text height.
        heights = [
            word["h"]
            for word in words
            if word.get("h", 0) > 0
        ]

        average_height = (
            sum(heights) / len(heights)
            if heights
            else 20
        )

        # Line tolerance.
        y_tolerance = max(
            8,
            int(
                average_height * 0.6
            ),
        )

        for word in sorted(
            words,
            key=lambda item: (
                item["y"],
                item["x"],
            ),
        ):

            placed = False

            center_y = (
                word["y"]
                + word["h"] / 2
            )

            for line in lines:

                if abs(
                    center_y
                    - line["center_y"]
                ) <= y_tolerance:

                    line["words"].append(
                        word
                    )

                    centers = [
                        item["y"]
                        + item["h"] / 2
                        for item in line[
                            "words"
                        ]
                    ]

                    line["center_y"] = (
                        sum(centers)
                        / len(centers)
                    )

                    placed = True
                    break

            if not placed:

                lines.append(
                    {
                        "center_y": center_y,
                        "words": [word],
                    }
                )

        # Sort lines vertically.
        lines.sort(
            key=lambda item: (
                item["center_y"]
            )
        )

        output_lines = []

        for line in lines:

            line_words = sorted(
                line["words"],
                key=lambda item: (
                    item["x"]
                ),
            )

            text = " ".join(
                item["text"]
                for item in line_words
            )

            text = re.sub(
                r"\s+",
                " ",
                text
            ).strip()

            if text:
                output_lines.append(
                    text
                )

        return "\n".join(
            output_lines
        )

    # =====================================================
    # OCR RESULT SCORE
    # =====================================================

    def score_ocr_result(
        self,
        result,
    ):
        """
        Score OCR result quality.

        Higher:
            confidence
            useful character count
            number of words

        Lower:
            excessive noise
        """

        if not result:
            return 0.0

        text = result.get(
            "text",
            ""
        )

        confidence = float(
            result.get(
                "confidence",
                0
            )
        )

        words = result.get(
            "words",
            []
        )

        if not text:
            return 0.0

        useful_chars = len(
            re.findall(
                r"[A-Za-z0-9]",
                text,
            )
        )

        word_count = len(
            words
        )

        # Don't allow length to dominate confidence.
        text_score = min(
            useful_chars / 20.0,
            10.0,
        )

        word_score = min(
            word_count / 10.0,
            5.0,
        )

        return (
            confidence
            + text_score
            + word_score
        )

    # =====================================================
    # NORMALIZE OCR LINE
    # =====================================================

    def normalize_ocr_line(
        self,
        line,
    ):
        """
        Normalize an OCR line while preserving useful
        punctuation.
        """

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

        line = re.sub(
            r"[ \t]+",
            " ",
            line,
        )

        return line.strip()

    # =====================================================
    # MERGE OCR TEXT
    # =====================================================

    def merge_text(
        self,
        tesseract,
        easy,
    ):
        """
        Merge Tesseract and EasyOCR results.

        Tesseract is preferred for:
            - line structure
            - field labels
            - punctuation
            - numbers

        EasyOCR can add:
            - difficult text
            - text missed by Tesseract
        """

        tess_text = (
            tesseract.get(
                "text",
                ""
            )
            if tesseract
            else ""
        )

        easy_text = (
            easy.get(
                "text",
                ""
            )
            if easy
            else ""
        )

        # -------------------------------------------------
        # If only Tesseract available
        # -------------------------------------------------

        if tess_text and not easy_text:
            return tess_text.strip()

        # -------------------------------------------------
        # If only EasyOCR available
        # -------------------------------------------------

        if easy_text and not tess_text:
            return easy_text.strip()

        # -------------------------------------------------
        # Nothing available
        # -------------------------------------------------

        if not tess_text and not easy_text:
            return ""

        # -------------------------------------------------
        # Start with Tesseract because its line structure
        # is generally better for field extraction.
        # -------------------------------------------------

        tess_lines = [
            self.normalize_ocr_line(
                line
            )
            for line in tess_text.splitlines()
        ]

        tess_lines = [
            line
            for line in tess_lines
            if line
        ]

        easy_lines = [
            self.normalize_ocr_line(
                line
            )
            for line in easy_text.splitlines()
        ]

        easy_lines = [
            line
            for line in easy_lines
            if line
        ]

        # -------------------------------------------------
        # Remove exact duplicate lines.
        # -------------------------------------------------

        merged = list(
            tess_lines
        )

        existing_normalized = {
            self.normalize_for_duplicate(
                line
            )
            for line in merged
        }

        for line in easy_lines:

            normalized = (
                self.normalize_for_duplicate(
                    line
                )
            )

            if not normalized:
                continue

            if normalized in existing_normalized:
                continue

            # Only add useful EasyOCR lines.
            if self.is_useful_ocr_line(
                line
            ):

                merged.append(
                    line
                )

                existing_normalized.add(
                    normalized
                )

        return "\n".join(
            merged
        )

    # =====================================================
    # DUPLICATE NORMALIZATION
    # =====================================================

    def normalize_for_duplicate(
        self,
        text,
    ):
        """
        Normalize text only for duplicate detection.
        """

        if not text:
            return ""

        text = str(text).upper()

        text = re.sub(
            r"[^A-Z0-9]+",
            "",
            text,
        )

        return text

    # =====================================================
    # USEFUL OCR LINE
    # =====================================================

    def is_useful_ocr_line(
        self,
        text,
    ):
        """
        Filter obvious OCR garbage.
        """

        if not text:
            return False

        alphanumeric = re.findall(
            r"[A-Za-z0-9]",
            text,
        )

        if len(alphanumeric) < 2:
            return False

        return True

    # =====================================================
    # MAIN EXTRACTION
    # =====================================================

    def extract_easyocr(self, image):
        if not OCR_USE_EASYOCR:
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
                    [OCR_LANGUAGE],
                    gpu=OCR_GPU,
                    verbose=False
                )

            results = self.easy_reader.readtext(
                image,
                detail=1,
                paragraph=False
            )

            words = []
            text_parts = []
            confidence_values = []

            for result in results:
                if len(result) < 3:
                    continue

                box, text, confidence = result

                text = str(text).strip()
                confidence = float(confidence) * 100

                if not text:
                    continue

                if confidence < OCR_MIN_CONFIDENCE:
                    continue

                xs = [int(point[0]) for point in box]
                ys = [int(point[1]) for point in box]

                x1 = min(xs)
                y1 = min(ys)
                x2 = max(xs)
                y2 = max(ys)

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

            return {
                "text": " ".join(text_parts),
                "words": words,
                "confidence": round(
                    sum(confidence_values) / len(confidence_values),
                    2
                ) if confidence_values else 0.0,
                "available": True,
            }

        except Exception as exc:
            return {
                "text": "",
                "words": [],
                "confidence": 0.0,
                "available": False,
                "error": str(exc),
            }
        
    def extract(
        self,
        path,
    ):
        """
        Complete OCR pipeline.

        IMPORTANT:
        OCR coordinates are returned in ORIGINAL IMAGE coordinates.
        This prevents highlight drift when OCR preprocessing upscales
        the image.
        """
        image = self.load_image(path)
        original_h, original_w = image.shape[:2]

        variants = self.preprocess_variants(image)
        tesseract_results = []

        for variant_name, processed in variants:
            result = self.extract_tesseract_multi(processed)
            if result.get("available") and result.get("text", "").strip():
                result["variant"] = variant_name
                result["_ocr_width"] = processed.shape[1]
                result["_ocr_height"] = processed.shape[0]
                tesseract_results.append(result)

        if tesseract_results:
            best_tesseract = max(
                tesseract_results,
                key=self.score_ocr_result,
            )
        else:
            best_tesseract = {
                "text": "",
                "words": [],
                "confidence": 0,
                "available": self.is_tesseract_available(),
            }

        # -------------------------------------------------
        # Map OCR coordinates back to ORIGINAL image.
        # -------------------------------------------------
        ocr_w = int(best_tesseract.get("_ocr_width", original_w) or original_w)
        ocr_h = int(best_tesseract.get("_ocr_height", original_h) or original_h)

        sx = original_w / max(1, ocr_w)
        sy = original_h / max(1, ocr_h)

        mapped_words = []
        for word in best_tesseract.get("words", []) or []:
            try:
                x = int(round(float(word.get("x", 0)) * sx))
                y = int(round(float(word.get("y", 0)) * sy))
                w = int(round(float(word.get("w", 0)) * sx))
                h = int(round(float(word.get("h", 0)) * sy))
            except (TypeError, ValueError):
                continue

            if w <= 0 or h <= 0:
                continue

            item = dict(word)
            item.update({
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "x2": min(original_w - 1, x + w),
                "y2": min(original_h - 1, y + h),
                "original_x": x,
                "original_y": y,
                "original_w": w,
                "original_h": h,
                "coordinate_space": "original",
                "scale_x": sx,
                "scale_y": sy,
            })
            mapped_words.append(item)

        # -------------------------------------------------
        # Stable physical reading order and line index.
        # -------------------------------------------------
        groups = {}
        for item in mapped_words:
            key = (
                item.get("block_num", 0),
                item.get("par_num", 0),
                item.get("line_num", 0),
            )
            groups.setdefault(key, []).append(item)

        ordered_groups = sorted(
            groups.values(),
            key=lambda g: (
                min(x.get("y", 0) for x in g),
                min(x.get("x", 0) for x in g),
            ),
        )

        for line_index, line_words in enumerate(ordered_groups):
            line_words.sort(key=lambda x: x.get("x", 0))
            for word_index, item in enumerate(line_words):
                item["line_index"] = line_index
                item["word_index"] = word_index

        mapped_words.sort(
            key=lambda x: (
                x.get("line_index", 0),
                x.get("x", 0),
            )
        )

        final_text = self.build_line_text(mapped_words)

        # EasyOCR is used as an additional confidence signal only.
        easy_result = self.extract_easyocr(self.upscale(image))

        confidences = []
        tc = float(best_tesseract.get("confidence", 0) or 0)
        ec = float(easy_result.get("confidence", 0) or 0)
        if tc > 0:
            confidences.append(tc)
        if ec > 0:
            confidences.append(ec)

        return {
            "text": final_text.strip(),
            "confidence": round(
                sum(confidences) / len(confidences)
                if confidences else 0.0,
                1,
            ),
            "words": mapped_words,
            "lines": [
                {
                    "line_index": i,
                    "text": " ".join(w["text"] for w in group),
                    "words": group,
                }
                for i, group in enumerate(ordered_groups)
            ],
            "coordinate_space": "original",
            "image_width": original_w,
            "image_height": original_h,
        }
