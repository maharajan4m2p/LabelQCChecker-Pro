"""
=========================================================
Label QC Checker Pro
Barcode Detection & Comparison Engine
=========================================================

Purpose:
    Detect, decode and compare linear barcodes on approval
    and sample labels.

Features:
    - OpenCV BarcodeDetector
    - Multiple preprocessing passes
    - Image upscaling
    - Grayscale processing
    - Contrast enhancement
    - Adaptive thresholding
    - Sharpening
    - Barcode coordinate detection
    - Barcode value normalization
    - Approval vs Sample comparison
    - Safe handling of undecodable barcodes

=========================================================
"""

from pathlib import Path
import re

import cv2
import numpy as np


class BarcodeChecker:
    """
    Detect and compare real linear barcodes.

    This class is intentionally conservative:
    if a barcode is visible but cannot be decoded,
    it will NOT automatically be treated as a PASS.
    """

    def __init__(self):
        """
        Initialize OpenCV BarcodeDetector.

        Some OpenCV installations may not provide
        BarcodeDetector. The application should continue
        running instead of crashing.
        """

        self.detector = None

        try:
            if hasattr(cv2, "barcode") and hasattr(
                cv2.barcode,
                "BarcodeDetector"
            ):
                self.detector = cv2.barcode.BarcodeDetector()

        except Exception:
            self.detector = None

    # =====================================================
    # NORMALIZE BARCODE VALUE
    # =====================================================

    def normalize_value(self, value):
        """
        Normalize a barcode value before comparison.

        Removes:
            - leading/trailing spaces
            - spaces inside the value
            - OCR-like line breaks
            - non-printing characters

        Example:

            'CTN 25006025'
                ->
            'CTN25006025'
        """

        if value is None:
            return ""

        value = str(value).strip().upper()

        # Remove line breaks/tabs
        value = re.sub(r"[\r\n\t]+", "", value)

        # Remove all spaces
        value = re.sub(r"\s+", "", value)

        # Keep only useful barcode characters
        value = re.sub(
            r"[^A-Z0-9\-._/]+",
            "",
            value
        )

        return value

    # =====================================================
    # READ IMAGE
    # =====================================================

    def _read_image(self, path):
        """
        Safely load image from disk.
        """

        try:
            image = cv2.imread(
                str(path),
                cv2.IMREAD_COLOR
            )

            return image

        except Exception:
            return None

    # =====================================================
    # CREATE PREPROCESSING VARIANTS
    # =====================================================

    def _preprocess_variants(self, image):
        """
        Generate multiple image versions.

        This is important because barcode detection can fail
        on the original photograph due to:

            - blur
            - low contrast
            - screen moire
            - shadows
            - perspective
            - small barcode size
        """

        variants = []

        if image is None:
            return variants

        # -------------------------------------------------
        # Original
        # -------------------------------------------------

        variants.append(
            ("original", image)
        )

        # -------------------------------------------------
        # Grayscale
        # -------------------------------------------------

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

        variants.append(
            ("gray", gray)
        )

        # -------------------------------------------------
        # Upscaled grayscale
        # -------------------------------------------------

        enlarged = cv2.resize(
            gray,
            None,
            fx=2.0,
            fy=2.0,
            interpolation=cv2.INTER_CUBIC
        )

        variants.append(
            ("upscaled", enlarged)
        )

        # -------------------------------------------------
        # CLAHE contrast enhancement
        # -------------------------------------------------

        clahe = cv2.createCLAHE(
            clipLimit=2.5,
            tileGridSize=(8, 8)
        )

        enhanced = clahe.apply(gray)

        variants.append(
            ("clahe", enhanced)
        )

        # -------------------------------------------------
        # Sharpen
        # -------------------------------------------------

        blurred = cv2.GaussianBlur(
            gray,
            (0, 0),
            2
        )

        sharpened = cv2.addWeighted(
            gray,
            1.7,
            blurred,
            -0.7,
            0
        )

        variants.append(
            ("sharpened", sharpened)
        )

        # -------------------------------------------------
        # OTSU threshold
        # -------------------------------------------------

        _, otsu = cv2.threshold(
            gray,
            0,
            255,
            cv2.THRESH_BINARY
            + cv2.THRESH_OTSU
        )

        variants.append(
            ("otsu", otsu)
        )

        # -------------------------------------------------
        # Adaptive threshold
        # -------------------------------------------------

        adaptive = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            11
        )

        variants.append(
            ("adaptive", adaptive)
        )

        return variants

    # =====================================================
    # EXTRACT DETECTOR RESULT
    # =====================================================

    def _parse_detector_result(self, result):
        """
        OpenCV versions can return different structures from
        BarcodeDetector.detectAndDecode().

        This function safely extracts:

            decoded value
            detected points
            type information
        """

        decoded_values = []
        points = None
        types = []

        if result is None:
            return decoded_values, points, types

        # -------------------------------------------------
        # Tuple result
        # -------------------------------------------------

        if isinstance(result, tuple):

            for item in result:

                if item is None:
                    continue

                # String result
                if isinstance(item, str):

                    text = item.strip()

                    if text:
                        decoded_values.append(text)

                # List / tuple of strings
                elif isinstance(item, (list, tuple)):

                    for sub_item in item:

                        if isinstance(
                            sub_item,
                            str
                        ):

                            text = sub_item.strip()

                            if text:
                                decoded_values.append(
                                    text
                                )

                # NumPy arrays
                elif isinstance(item, np.ndarray):

                    if item.size == 0:
                        continue

                    # Barcode corner points
                    if (
                        item.dtype.kind
                        in ("f", "i", "u")
                        and item.ndim >= 2
                    ):
                        points = item

        # -------------------------------------------------
        # Single string result
        # -------------------------------------------------

        elif isinstance(result, str):

            if result.strip():
                decoded_values.append(
                    result.strip()
                )

        return decoded_values, points, types

    # =====================================================
    # DETECT USING ONE IMAGE
    # =====================================================

    def _detect_single(self, image):
        """
        Run BarcodeDetector on one image.
        """

        if self.detector is None:
            return {
                "decoded": "",
                "detected": False,
                "points": None,
                "format": "",
            }

        try:

            result = self.detector.detectAndDecode(
                image
            )

            decoded_values, points, types = (
                self._parse_detector_result(result)
            )

            decoded = ""

            if decoded_values:

                # Use the first useful decoded value
                for value in decoded_values:

                    value = str(value).strip()

                    if value:
                        decoded = value
                        break

            detected = (
                points is not None
                and np.size(points) > 0
            )

            # If decoder returned a value, barcode
            # definitely exists.
            if decoded:
                detected = True

            return {
                "decoded": decoded,
                "detected": detected,
                "points": points,
                "format": (
                    types[0]
                    if types
                    else ""
                ),
            }

        except Exception:
            return {
                "decoded": "",
                "detected": False,
                "points": None,
                "format": "",
            }

    # =====================================================
    # DETECT BARCODE
    # =====================================================

    def detect(self, path):
        """
        Detect a barcode from an image.

        Multiple preprocessing variants are attempted.

        Returns:

            {
                detected,
                value,
                status,
                points,
                method
            }
        """

        image = self._read_image(path)

        if image is None:
            return {
                "detected": False,
                "value": "",
                "status": "ERROR",
                "points": None,
                "method": "",
                "error": (
                    "Unable to read image."
                ),
            }

        if self.detector is None:
            return {
                "detected": False,
                "value": "",
                "status": "NOT_CHECKED",
                "points": None,
                "method": "",
                "error": (
                    "OpenCV BarcodeDetector "
                    "is unavailable."
                ),
            }

        variants = self._preprocess_variants(
            image
        )

        detected_without_value = None

        # -------------------------------------------------
        # Try every preprocessing version
        # -------------------------------------------------

        for method, variant in variants:

            result = self._detect_single(
                variant
            )

            # ---------------------------------------------
            # Successfully decoded
            # ---------------------------------------------

            if result["decoded"]:

                normalized = (
                    self.normalize_value(
                        result["decoded"]
                    )
                )

                return {
                    "detected": True,
                    "value": normalized,
                    "raw_value": result[
                        "decoded"
                    ],
                    "status": "DETECTED",
                    "points": result[
                        "points"
                    ],
                    "method": method,
                }

            # ---------------------------------------------
            # Barcode detected but not decoded
            # ---------------------------------------------

            if result["detected"]:

                if detected_without_value is None:

                    detected_without_value = {
                        "detected": True,
                        "value": "",
                        "raw_value": "",
                        "status": (
                            "DETECTED_NOT_DECODED"
                        ),
                        "points": result[
                            "points"
                        ],
                        "method": method,
                    }

        # -------------------------------------------------
        # Barcode found but value unavailable
        # -------------------------------------------------

        if detected_without_value is not None:

            return detected_without_value

        # -------------------------------------------------
        # Nothing found
        # -------------------------------------------------

        return {
            "detected": False,
            "value": "",
            "raw_value": "",
            "status": "NOT_FOUND",
            "points": None,
            "method": "",
        }

    # =====================================================
    # COMPARE BARCODE VALUES
    # =====================================================

    def compare(self, approval_path, sample_path):
        """
        Compare barcode information between:

            Approval Label
            Sample Label
        """

        approval = self.detect(
            approval_path
        )

        sample = self.detect(
            sample_path
        )

        # =================================================
        # ERROR / UNAVAILABLE
        # =================================================

        if (
            approval["status"]
            in ("ERROR", "NOT_CHECKED")
            or
            sample["status"]
            in ("ERROR", "NOT_CHECKED")
        ):

            return {
                "status": "NOT_CHECKED",

                "approval_value": approval.get(
                    "value",
                    ""
                ),

                "sample_value": sample.get(
                    "value",
                    ""
                ),

                "approval_detected": approval.get(
                    "detected",
                    False
                ),

                "sample_detected": sample.get(
                    "detected",
                    False
                ),

                "approval_points": approval.get(
                    "points"
                ),

                "sample_points": sample.get(
                    "points"
                ),

                "message": (
                    "Barcode detector is "
                    "unavailable or returned "
                    "an error."
                ),
            }

        # =================================================
        # NEITHER LABEL HAS BARCODE
        # =================================================

        if (
            not approval["detected"]
            and
            not sample["detected"]
        ):

            return {
                "status": "NOT_FOUND",

                "approval_value": "",

                "sample_value": "",

                "approval_detected": False,

                "sample_detected": False,

                "approval_points": None,

                "sample_points": None,

                "message": (
                    "No linear barcode detected "
                    "on either label."
                ),
            }

        # =================================================
        # APPROVAL HAS BARCODE
        # SAMPLE DOES NOT
        # =================================================

        if (
            approval["detected"]
            and
            not sample["detected"]
        ):

            return {
                "status": "FAIL",

                "approval_value": approval.get(
                    "value",
                    ""
                ),

                "sample_value": "",

                "approval_detected": True,

                "sample_detected": False,

                "approval_points": approval.get(
                    "points"
                ),

                "sample_points": None,

                "message": (
                    "Barcode exists on the "
                    "approval label but is "
                    "missing from the sample."
                ),
            }

        # =================================================
        # SAMPLE HAS BARCODE
        # APPROVAL DOES NOT
        # =================================================

        if (
            not approval["detected"]
            and
            sample["detected"]
        ):

            return {
                "status": "FAIL",

                "approval_value": "",

                "sample_value": sample.get(
                    "value",
                    ""
                ),

                "approval_detected": False,

                "sample_detected": True,

                "approval_points": None,

                "sample_points": sample.get(
                    "points"
                ),

                "message": (
                    "Barcode exists on the "
                    "sample label but is "
                    "missing from the approval."
                ),
            }

        # =================================================
        # BOTH DETECTED
        # BUT VALUES NOT DECODABLE
        # =================================================

        approval_value = self.normalize_value(
            approval.get(
                "value",
                ""
            )
        )

        sample_value = self.normalize_value(
            sample.get(
                "value",
                ""
            )
        )

        if (
            not approval_value
            and
            not sample_value
        ):

            return {
                "status": (
                    "DETECTED_NOT_DECODED"
                ),

                "approval_value": "",

                "sample_value": "",

                "approval_detected": True,

                "sample_detected": True,

                "approval_points": approval.get(
                    "points"
                ),

                "sample_points": sample.get(
                    "points"
                ),

                "message": (
                    "Barcode detected on both "
                    "labels, but neither "
                    "barcode value could "
                    "be decoded."
                ),
            }

        # =================================================
        # ONLY APPROVAL VALUE DECODED
        # =================================================
        if (
            approval_value
            and
            not sample_value
        ):

            return {
                "status": "NOT_CHECKED",

                "approval_value": (
                    approval_value
                ),

                "sample_value": "",

                "approval_detected": True,

                "sample_detected": True,

                "approval_points": approval.get(
                    "points"
                ),

                "sample_points": sample.get(
                    "points"
                ),

                "message": (
                    "Approval barcode decoded, "
                    "but sample barcode could "
                    "not be decoded. Value "
                    "comparison cannot be "
                    "confirmed."
                ),
            }

        # =================================================
        # ONLY SAMPLE VALUE DECODED
        # =================================================

        if (
            not approval_value
            and
            sample_value
        ):

            return {
                "status": "NOT_CHECKED",

                "approval_value": "",

                "sample_value": (
                    sample_value
                ),

                "approval_detected": True,

                "sample_detected": True,

                "approval_points": approval.get(
                    "points"
                ),

                "sample_points": sample.get(
                    "points"
                ),

                "message": (
                    "Sample barcode decoded, "
                    "but approval barcode "
                    "could not be decoded. "
                    "Value comparison cannot "
                    "be confirmed."
                ),
            }

        # =================================================
        # BOTH VALUES DECODED
        # =================================================

        if approval_value == sample_value:

            return {
                "status": "PASS",

                "approval_value": (
                    approval_value
                ),

                "sample_value": (
                    sample_value
                ),

                "approval_detected": True,

                "sample_detected": True,

                "approval_points": approval.get(
                    "points"
                ),

                "sample_points": sample.get(
                    "points"
                ),

                "message": (
                    "Barcode detected and "
                    "values match exactly."
                ),
            }

        # =================================================
        # VALUES ARE DIFFERENT
        # =================================================

        return {
            "status": "FAIL",

            "approval_value": (
                approval_value
            ),

            "sample_value": (
                sample_value
            ),

            "approval_detected": True,

            "sample_detected": True,

            "approval_points": approval.get(
                "points"
            ),

            "sample_points": sample.get(
                "points"
            ),

            "message": (
                "Barcode detected on both "
                "labels, but barcode "
                "values are different."
            ),
        }