"""
=========================================================
Label QC Checker Pro
Global Configuration
=========================================================

Central configuration for:

    - Flask
    - File uploads
    - OCR
    - EasyOCR
    - Tesseract
    - Field detection
    - Text comparison
    - Logo comparison
    - Barcode comparison
    - Image highlighting
=========================================================
"""

from pathlib import Path


# =========================================================
# PROJECT DIRECTORIES
# =========================================================

BASE_DIR = Path(
    __file__
).resolve().parent

UPLOAD_DIR = (
    BASE_DIR
    / "uploads"
)

OUTPUT_DIR = (
    BASE_DIR
    / "outputs"
)

# Create directories automatically.
UPLOAD_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =========================================================
# FILE UPLOAD CONFIGURATION
# =========================================================

ALLOWED_EXTENSIONS = {
    "png",
    "jpg",
    "jpeg",
    "webp",
    "pdf",
}

# ---------------------------------------------------------
# Maximum upload size:
#
# 25 MB
# ---------------------------------------------------------

MAX_CONTENT_LENGTH = (
    50 * 1024 * 1024
)


# =========================================================
# OCR CONFIGURATION
# =========================================================

# Tesseract language.
OCR_LANGUAGE = "eng"

# ---------------------------------------------------------
# Tesseract Page Segmentation Mode
#
# 6 = Assume a uniform block of text
#
# Good general starting point for labels.
# ---------------------------------------------------------

OCR_PSM = 6

# ---------------------------------------------------------
# Minimum OCR confidence.
#
# Words below this confidence are ignored.
# ---------------------------------------------------------

OCR_MIN_CONFIDENCE = 25


# =========================================================
# EASY OCR
# =========================================================

# Set True only when EasyOCR is installed and you want
# the additional OCR engine enabled.
OCR_USE_EASYOCR = False

# Use GPU for EasyOCR.
#
# False is safer for normal Windows/CPU installations.
#
# Set True only when CUDA/PyTorch GPU support is properly
# installed.
# ---------------------------------------------------------

OCR_GPU = False


# =========================================================
# OCR IMAGE PROCESSING
# =========================================================

# Minimum image width used by OCR preprocessing.
#
# Small label images are enlarged before OCR.
# ---------------------------------------------------------

OCR_MIN_WIDTH = 1600

# Minimum image height used by OCR preprocessing.
# ---------------------------------------------------------

OCR_MIN_HEIGHT = 1200

# Upscale factor used for OCR.
# ---------------------------------------------------------

OCR_SCALE_FACTOR = 2.0

# Maximum image width after upscaling.
# ---------------------------------------------------------

OCR_MAX_WIDTH = 3200

# Maximum image height after upscaling.
# ---------------------------------------------------------

OCR_MAX_HEIGHT = 3200


# =========================================================
# OCR PREPROCESSING
# =========================================================

# CLAHE contrast enhancement.
# ---------------------------------------------------------

OCR_CLAHE_CLIP_LIMIT = 2.0

OCR_CLAHE_TILE_GRID_SIZE = (
    8,
    8,
)

# Gaussian blur kernel.
# ---------------------------------------------------------

OCR_BLUR_KERNEL = (
    3,
    3,
)

# Adaptive threshold block size.
# Must be an odd number.
# ---------------------------------------------------------

OCR_THRESHOLD_BLOCK_SIZE = 31

# Adaptive threshold constant.
# ---------------------------------------------------------

OCR_THRESHOLD_C = 11


# =========================================================
# OCR PARALLEL PROCESSING
# =========================================================

# Enable parallel OCR engines when multiple engines are
# enabled.
# ---------------------------------------------------------

OCR_PARALLEL = True

# Maximum worker threads.
# ---------------------------------------------------------

OCR_MAX_WORKERS = 3

# Merge OCR results from multiple engines.
# ---------------------------------------------------------

OCR_MERGE_RESULTS = True

# Similarity threshold used when merging OCR words.
# ---------------------------------------------------------

OCR_MERGE_THRESHOLD = 0.80


# =========================================================
# TESSERACT CONFIGURATION
# =========================================================

# Tesseract OEM.
#
# 3 = Default engine mode.
# ---------------------------------------------------------

OCR_OEM = 3

# Tesseract PSM.
# Kept separately so OCR engine can access it directly.
# ---------------------------------------------------------

TESSERACT_PSM = OCR_PSM

# Tesseract minimum confidence.
# ---------------------------------------------------------

TESSERACT_MIN_CONFIDENCE = OCR_MIN_CONFIDENCE


# =========================================================
# FIELD COMPARISON
# =========================================================

# Minimum similarity for a field to be considered MATCH.
#
# 0.88 = 88%
# ---------------------------------------------------------

FIELD_MATCH_THRESHOLD = 0.88

# Word-level comparison threshold.
# ---------------------------------------------------------

WORD_MATCH_THRESHOLD = 0.82

# Character-level similarity threshold.
# ---------------------------------------------------------

TEXT_MATCH_THRESHOLD = 0.90


# =========================================================
# OVERALL QC SCORE
# =========================================================

# Overall PASS threshold.
#
# 0.90 = 90%
# ---------------------------------------------------------

OVERALL_PASS_THRESHOLD = 0.90


# =========================================================
# SCORE WEIGHTS
# =========================================================

# Main field/text comparison.
# ---------------------------------------------------------

FIELD_SCORE_WEIGHT = 0.85

# Logo comparison.
# ---------------------------------------------------------

LOGO_SCORE_WEIGHT = 0.10

# Barcode comparison.
# ---------------------------------------------------------

BARCODE_SCORE_WEIGHT = 0.05


# =========================================================
# LOGO CONFIGURATION
# =========================================================

# Minimum logo score required for PASS.
# ---------------------------------------------------------

LOGO_MATCH_THRESHOLD = 0.70

# Minimum visual edge density.
# ---------------------------------------------------------

LOGO_MIN_EDGE_RATIO = 0.015

# Minimum grayscale variation.
# ---------------------------------------------------------

LOGO_MIN_STD = 8.0

# Final structural logo threshold.
# ---------------------------------------------------------

LOGO_STRUCTURE_THRESHOLD = 0.68


# =========================================================
# LOGO REGION
# =========================================================

# Default logo crop coordinates as percentages.
#
# Example:
#
# top    = 2%
# bottom = 30%
# left   = 2%
# right  = 45%
#
# ---------------------------------------------------------

LOGO_TOP = 0.02

LOGO_BOTTOM = 0.30

LOGO_LEFT = 0.02

LOGO_RIGHT = 0.45


# =========================================================
# BARCODE CONFIGURATION
# =========================================================

# Barcode is optional.
#
# False:
#     A missing barcode does not automatically cause
#     overall failure.
#
# True:
#     Barcode should be present.
# ---------------------------------------------------------

BARCODE_REQUIRED = False

# Barcode matching threshold.
# ---------------------------------------------------------

BARCODE_MATCH_THRESHOLD = 1.00


# =========================================================
# EXPECTED BUSINESS FIELDS
# =========================================================

"""
Aliases are used by FieldDetector.

Example:

    PO NO:
        PO NO
        PO NUMBER
        P.O. NO
        PONO
        PURCHASE ORDER

If your company label uses another name, add it here.
"""

FIELD_ALIASES = {

    # -----------------------------------------------------
    # BUYER
    # -----------------------------------------------------

    "BUYER": [
        "BUYER",
        "BUYER NAME",
        "CUSTOMER",
        "CUSTOMER NAME",
    ],

    # -----------------------------------------------------
    # BUYER CODE
    # -----------------------------------------------------

    "BUYER CODE": [
        "BUYER CODE",
        "BUYERCODE",
        "BUYER ID",
        "CUSTOMER CODE",
        "CUSTOMER ID",
    ],

    # -----------------------------------------------------
    # VENDOR
    # -----------------------------------------------------

    "VENDOR": [
        "VENDOR",
        "SUPPLIER",
        "VENDOR NAME",
        "SUPPLIER NAME",
    ],

    # -----------------------------------------------------
    # PO NUMBER
    # -----------------------------------------------------

    "PO NO": [
        "PO NO",
        "PO NUMBER",
        "P.O. NO",
        "P.O NO",
        "P.O. NUMBER",
        "PONO",
        "PURCHASE ORDER",
        "PURCHASE ORDER NO",
        "PURCHASE ORDER NUMBER",
    ],

    # -----------------------------------------------------
    # STYLE NUMBER
    # -----------------------------------------------------

    "STYLE NO": [
        "STYLE NO",
        "STYLE NUMBER",
        "STYLE",
        "STYLE CODE",
        "STY NO",
        "STY NUMBER",
    ],

    # -----------------------------------------------------
    # DESCRIPTION
    # -----------------------------------------------------

    "DESCRIPTION": [
        "DESCRIPTION",
        "DESC",
        "ITEM DESCRIPTION",
        "PRODUCT DESCRIPTION",
    ],

    # -----------------------------------------------------
    # COLOR
    # -----------------------------------------------------

    "COLOR": [
        "COLOR",
        "COLOUR",
        "COLOR NAME",
        "COLOUR NAME",
    ],

    # -----------------------------------------------------
    # SIZE
    # -----------------------------------------------------

    "SIZE": [
        "SIZE",
        "SIZE NAME",
        "SIZE CODE",
    ],

    # -----------------------------------------------------
    # QUANTITY
    # -----------------------------------------------------

    "QTY": [
        "QTY",
        "QUANTITY",
        "PCS",
        "PIECES",
        "QTY/PCS",
        "QTY PCS",
        "TOTAL PCS",
    ],

    # -----------------------------------------------------
    # TOTAL QUANTITY
    # -----------------------------------------------------

    "TOTAL QTY": [
        "TOTAL QTY",
        "TOTAL QUANTITY",
        "TOTAL PCS",
        "TOTAL PIECES",
    ],

    # -----------------------------------------------------
    # GROSS WEIGHT
    # -----------------------------------------------------

    "G.W.": [
        "G.W.",
        "G.W",
        "GW",
        "GROSS WEIGHT",
        "GROSS WT",
    ],

    # -----------------------------------------------------
    # NET WEIGHT
    # -----------------------------------------------------

    "N.W.": [
        "N.W.",
        "N.W",
        "NW",
        "NET WEIGHT",
        "NET WT",
    ],

    # -----------------------------------------------------
    # MEASUREMENT
    # -----------------------------------------------------

    "MEASUREMENT": [
        "MEASUREMENT",
        "MEASUREMENTS",
        "MEASURE",
        "DIMENSION",
        "DIMENSIONS",
    ],

    # -----------------------------------------------------
    # VOLUME
    # -----------------------------------------------------

    "VOLUME": [
        "VOLUME",
        "CBM",
        "CUBIC METER",
        "CUBIC METRE",
    ],

    # -----------------------------------------------------
    # CARTON NUMBER
    # -----------------------------------------------------

    "CARTON NO": [
        "CARTON NO",
        "CARTON NUMBER",
        "CARTON",
        "CTN NO",
        "CTN NUMBER",
        "CARTON #",
    ],

    # -----------------------------------------------------
    # COUNTRY OF ORIGIN
    # -----------------------------------------------------

    "COUNTRY OF ORIGIN": [
        "COUNTRY OF ORIGIN",
        "ORIGIN",
        "MADE IN",
        "COUNTRY",
    ],

    # -----------------------------------------------------
    # DESTINATION
    # -----------------------------------------------------

    "DESTINATION": [
        "DESTINATION",
        "DEST",
        "FINAL DESTINATION",
    ],

    # -----------------------------------------------------
    # PORT OF LOADING
    # -----------------------------------------------------

    "PORT OF LOADING": [
        "PORT OF LOADING",
        "LOADING PORT",
        "POL",
    ],

    # -----------------------------------------------------
    # PORT OF DISCHARGE
    # -----------------------------------------------------

    "PORT OF DISCHARGE": [
        "PORT OF DISCHARGE",
        "DISCHARGE PORT",
        "POD",
    ],

    # -----------------------------------------------------
    # SHIPMENT MODE
    # -----------------------------------------------------

    "SHIPMENT MODE": [
        "SHIPMENT MODE",
        "MODE OF SHIPMENT",
        "MODE",
        "SHIP MODE",
    ],

    # -----------------------------------------------------
    # ETD
    # -----------------------------------------------------

    "ETD": [
        "ETD",
        "ESTIMATED TIME OF DEPARTURE",
        "EST. TIME OF DEPARTURE",
    ],

    # -----------------------------------------------------
    # ETA
    # -----------------------------------------------------

    "ETA": [
        "ETA",
        "ESTIMATED TIME OF ARRIVAL",
        "EST. TIME OF ARRIVAL",
    ],
}


# =========================================================
# FIELD NAMES
# =========================================================

FIELD_NAMES = list(
    FIELD_ALIASES.keys()
)


# =========================================================
# TEXT NORMALIZATION
# =========================================================

# Characters removed/replaced during comparison.
NORMALIZE_REMOVE_CHARS = (
    ":;|"
)

# Additional OCR punctuation normalization.
NORMALIZE_REPLACEMENTS = {
    "—": "-",
    "–": "-",
    "−": "-",
    "“": '"',
    "”": '"',
    "‘": "'",
    "’": "'",
}


# =========================================================
# OCR COMMON CHARACTER CORRECTIONS
# =========================================================

"""
Common OCR mistakes.

These should NOT blindly replace every character in every
field because values such as PO numbers can legitimately
contain O/0 and I/1.

Therefore these are available for specialized matching,
rather than globally changing OCR text.
"""

OCR_CHARACTER_CONFUSIONS = {
    "O": "0",
    "I": "1",
    "L": "1",
    "S": "5",
    "B": "8",
    "Z": "2",
}


# =========================================================
# HIGHLIGHT CONFIGURATION
# =========================================================

# Padding around detected word.
HIGHLIGHT_PADDING = 6

# Thickness of change rectangle.
HIGHLIGHT_THICKNESS = 3

# Maximum text shown inside annotation.
HIGHLIGHT_MAX_TEXT_LENGTH = 70

# Annotation font scale.
HIGHLIGHT_FONT_SCALE = 0.55

# Annotation thickness.
HIGHLIGHT_FONT_THICKNESS = 2


# =========================================================
# OUTPUT IMAGE CONFIGURATION
# =========================================================

OUTPUT_IMAGE_EXTENSION = ".jpg"

OUTPUT_IMAGE_QUALITY = 95


# =========================================================
# PDF CONFIGURATION
# =========================================================

PDF_REPORT_TITLE = (
    "LABEL QC CHECKER PRO"
)

PDF_REPORT_SUBTITLE = (
    "Approval Label vs Sample Label "
    "Comparison Report"
)


# =========================================================
# APPLICATION CONFIGURATION
# =========================================================

FLASK_HOST = "0.0.0.0"

FLASK_PORT = 5000

FLASK_DEBUG = True


# =========================================================
# RESULT STATUS VALUES
# =========================================================

STATUS_MATCH = "MATCH"

STATUS_MISMATCH = "MISMATCH"

STATUS_MISSING = "MISSING"

STATUS_EXTRA = "EXTRA"

STATUS_NOT_CHECKED = "NOT_CHECKED"

STATUS_NOT_FOUND = "NOT_FOUND"

STATUS_PASS = "PASS"

STATUS_FAIL = "FAIL"

STATUS_ERROR = "ERROR"


# =========================================================
# VALIDATION
# =========================================================

def validate_config():
    """
    Validate important configuration values.

    Raises ValueError when an invalid configuration is
    detected.
    """

    # -----------------------------------------------------
    # Thresholds
    # -----------------------------------------------------

    thresholds = {
        "FIELD_MATCH_THRESHOLD":
            FIELD_MATCH_THRESHOLD,

        "WORD_MATCH_THRESHOLD":
            WORD_MATCH_THRESHOLD,

        "TEXT_MATCH_THRESHOLD":
            TEXT_MATCH_THRESHOLD,

        "OVERALL_PASS_THRESHOLD":
            OVERALL_PASS_THRESHOLD,

        "LOGO_MATCH_THRESHOLD":
            LOGO_MATCH_THRESHOLD,
    }

    for name, value in thresholds.items():

        if not 0 <= float(value) <= 1:

            raise ValueError(
                f"{name} must be between "
                f"0 and 1. Current value: {value}"
            )

    # -----------------------------------------------------
    # OCR confidence
    # -----------------------------------------------------

    if not (
        0 <= OCR_MIN_CONFIDENCE <= 100
    ):

        raise ValueError(
            "OCR_MIN_CONFIDENCE must be "
            "between 0 and 100."
        )

    # -----------------------------------------------------
    # OCR PSM
    # -----------------------------------------------------

    if OCR_PSM < 0:

        raise ValueError(
            "OCR_PSM cannot be negative."
        )

    # -----------------------------------------------------
    # Upload size
    # -----------------------------------------------------

    if MAX_CONTENT_LENGTH <= 0:

        raise ValueError(
            "MAX_CONTENT_LENGTH must "
            "be greater than zero."
        )

    # -----------------------------------------------------
    # OCR scale
    # -----------------------------------------------------

    if OCR_SCALE_FACTOR <= 0:

        raise ValueError(
            "OCR_SCALE_FACTOR must "
            "be greater than zero."
        )

    return True


# =========================================================
# RUN CONFIG VALIDATION
# =========================================================

validate_config()