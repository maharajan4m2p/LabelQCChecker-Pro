"""
=========================================================
Label QC Checker Pro
PDF Report Generator
=========================================================

Purpose:
    Generate a professional PDF report from the label
    comparison result.

Includes:
    - Approval label information
    - Sample label information
    - Overall QC score
    - PASS / FAIL
    - Field comparison
    - MISMATCH details
    - MISSING fields
    - EXTRA fields
    - Logo result
    - Barcode result
    - OCR confidence
    - Multiple sample labels
    - Page breaks
    - Safe handling of missing data
=========================================================
"""

from pathlib import Path
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import (
    getSampleStyleSheet,
    ParagraphStyle,
)
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
    KeepTogether,
)


# =========================================================
# SAFE VALUE
# =========================================================

def safe_value(
    value,
    default="",
):
    """
    Safely convert a value to a printable string.
    """

    if value is None:
        return default

    value = str(
        value
    ).strip()

    return (
        value
        if value
        else default
    )


# =========================================================
# ESCAPE PDF TEXT
# =========================================================

def pdf_text(
    value,
    default="-",
):
    """
    Convert text into safe ReportLab Paragraph content.

    Basic HTML-sensitive characters are escaped so OCR text
    cannot break the PDF layout.
    """

    value = safe_value(
        value,
        default,
    )

    value = (
        value
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )

    # Preserve simple line breaks.
    value = value.replace(
        "\n",
        "<br/>",
    )

    return value


# =========================================================
# STATUS COLOR
# =========================================================

def status_color(
    status,
):
    """
    Return a PDF color based on comparison status.
    """

    status = safe_value(
        status
    ).upper()

    if status == "PASS":
        return colors.HexColor(
            "#198754"
        )

    if status == "MATCH":
        return colors.HexColor(
            "#198754"
        )

    if status == "MISMATCH":
        return colors.HexColor(
            "#DC3545"
        )

    if status == "FAIL":
        return colors.HexColor(
            "#DC3545"
        )

    if status == "MISSING":
        return colors.HexColor(
            "#FD7E14"
        )

    if status == "EXTRA":
        return colors.HexColor(
            "#0D6EFD"
        )

    if status == "NOT_CHECKED":
        return colors.HexColor(
            "#6C757D"
        )

    if status == "NOT_FOUND":
        return colors.HexColor(
            "#6C757D"
        )

    return colors.HexColor(
        "#6C757D"
    )


# =========================================================
# BUILD PDF REPORT
# =========================================================

def build_pdf_report(
    result,
    output_dir,
):
    """
    Generate a complete PDF comparison report.

    Parameters:

        result:
            Dictionary returned by compare_labels()

        output_dir:
            Folder where the PDF should be saved.

    Returns:

        String filesystem path of generated PDF.
    """

    # =====================================================
    # VALIDATE RESULT
    # =====================================================

    if not result:
        raise ValueError(
            "No comparison result was supplied."
        )

    result = dict(
        result
    )

    approval = result.get(
        "approval",
        {},
    ) or {}

    samples = result.get(
        "samples",
        [],
    ) or []

    # =====================================================
    # OUTPUT DIRECTORY
    # =====================================================

    output_dir = Path(
        output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # =====================================================
    # PDF FILENAME
    # =====================================================

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    filename = (
        "label_qc_report_"
        f"{timestamp}.pdf"
    )

    pdf_path = (
        output_dir
        / filename
    )

    # =====================================================
    # DOCUMENT
    # =====================================================

    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=landscape(A4),
        rightMargin=10 * mm,
        leftMargin=10 * mm,
        topMargin=10 * mm,
        bottomMargin=10 * mm,
        title=(
            "Label QC Checker "
            "Comparison Report"
        ),
        author=(
            "Label QC Checker Pro"
        ),
    )

    # =====================================================
    # STYLES
    # =====================================================

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "QC_Title",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        alignment=TA_CENTER,
        spaceAfter=8,
    )

    subtitle_style = ParagraphStyle(
        "QC_Subtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=12,
        alignment=TA_CENTER,
        textColor=colors.HexColor(
            "#555555"
        ),
        spaceAfter=10,
    )

    heading_style = ParagraphStyle(
        "QC_Heading",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=15,
        textColor=colors.HexColor(
            "#123B8A"
        ),
        spaceBefore=5,
        spaceAfter=6,
    )

    normal_style = ParagraphStyle(
        "QC_Normal",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=11,
    )

    small_style = ParagraphStyle(
        "QC_Small",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=7,
        leading=9,
    )

    center_style = ParagraphStyle(
        "QC_Center",
        parent=normal_style,
        alignment=TA_CENTER,
    )

    # =====================================================
    # STORY
    # =====================================================

    story = []

    # =====================================================
    # TITLE
    # =====================================================

    story.append(
        Paragraph(
            "LABEL QC CHECKER PRO",
            title_style,
        )
    )

    story.append(
        Paragraph(
            "Approval Label vs Sample Label "
            "Comparison Report",
            subtitle_style,
        )
    )

    # =====================================================
    # REPORT INFORMATION
    # =====================================================

    approval_filename = safe_value(
        approval.get(
            "filename"
        ),
        "-",
    )

    report_info = [
        [
            Paragraph(
                "<b>Approval Label</b>",
                normal_style,
            ),
            Paragraph(
                pdf_text(
                    approval_filename
                ),
                normal_style,
            ),
        ],
        [
            Paragraph(
                "<b>Generated</b>",
                normal_style,
            ),
            Paragraph(
                datetime.now().strftime(
                    "%d-%m-%Y %H:%M:%S"
                ),
                normal_style,
            ),
        ],
        [
            Paragraph(
                "<b>Samples Compared</b>",
                normal_style,
            ),
            Paragraph(
                str(
                    len(samples)
                ),
                normal_style,
            ),
        ],
    ]

    report_info_table = Table(
        report_info,
        colWidths=[
            45 * mm,
            220 * mm,
        ],
    )

    report_info_table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (0, -1),
                    colors.HexColor(
                        "#E9ECEF"
                    ),
                ),
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.4,
                    colors.HexColor(
                        "#BBBBBB"
                    ),
                ),
                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "MIDDLE",
                ),
                (
                    "LEFTPADDING",
                    (0, 0),
                    (-1, -1),
                    6,
                ),
                (
                    "RIGHTPADDING",
                    (0, 0),
                    (-1, -1),
                    6,
                ),
                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    5,
                ),
                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    5,
                ),
            ]
        )
    )

    story.append(
        report_info_table
    )

    story.append(
        Spacer(
            1,
            10,
        )
    )

    # =====================================================
    # NO SAMPLES
    # =====================================================

    if not samples:

        story.append(
            Paragraph(
                "No sample labels were available "
                "for comparison.",
                heading_style,
            )
        )

        doc.build(
            story
        )

        return str(
            pdf_path
        )

    # =====================================================
    # EACH SAMPLE
    # =====================================================

    for sample_index, sample in enumerate(
        samples,
        start=1,
    ):

        sample = sample or {}

        # -------------------------------------------------
        # Sample details
        # -------------------------------------------------

        sample_filename = safe_value(
            sample.get(
                "filename"
            ),
            "-",
        )

        score = sample.get(
            "score",
            0,
        )

        status = safe_value(
            sample.get(
                "status"
            ),
            "FAIL",
        ).upper()

        # -------------------------------------------------
        # Sample heading
        # -------------------------------------------------

        story.append(
            Paragraph(
                f"Sample {sample_index}: "
                f"{pdf_text(sample_filename)}",
                heading_style,
            )
        )

        # -------------------------------------------------
        # Summary cards/table
        # -------------------------------------------------

        field_result = (
            sample.get(
                "field_result",
                {},
            )
            or {}
        )

        matched = field_result.get(
            "matched",
            sample.get(
                "matched",
                0,
            ),
        )

        mismatched = field_result.get(
            "mismatched",
            sample.get(
                "mismatched",
                0,
            ),
        )

        missing = field_result.get(
            "missing",
            sample.get(
                "missing",
                0,
            ),
        )

        extra = field_result.get(
            "extra",
            sample.get(
                "extra",
                0,
            ),
        )

        not_checked = field_result.get(
            "not_checked",
            sample.get(
                "not_checked",
                0,
            ),
        )

        total = field_result.get(
            "total",
            sample.get(
                "total",
                0,
            ),
        )

        ocr_confidence = sample.get(
            "ocr_confidence",
            0,
        )

        summary_rows = [
            [
                Paragraph(
                    "<b>Overall Score</b>",
                    center_style,
                ),
                Paragraph(
                    "<b>Result</b>",
                    center_style,
                ),
                Paragraph(
                    "<b>Matched</b>",
                    center_style,
                ),
                Paragraph(
                    "<b>Mismatched</b>",
                    center_style,
                ),
                Paragraph(
                    "<b>Missing</b>",
                    center_style,
                ),
                Paragraph(
                    "<b>Extra</b>",
                    center_style,
                ),
                Paragraph(
                    "<b>Not Checked</b>",
                    center_style,
                ),
                Paragraph(
                    "<b>OCR Confidence</b>",
                    center_style,
                ),
            ],
            [
                Paragraph(
                    f"<b>{pdf_text(score)}%</b>",
                    center_style,
                ),
                Paragraph(
                    f"<b>{pdf_text(status)}</b>",
                    center_style,
                ),
                Paragraph(
                    str(matched),
                    center_style,
                ),
                Paragraph(
                    str(mismatched),
                    center_style,
                ),
                Paragraph(
                    str(missing),
                    center_style,
                ),
                Paragraph(
                    str(extra),
                    center_style,
                ),
                Paragraph(
                    str(not_checked),
                    center_style,
                ),
                Paragraph(
                    f"{pdf_text(ocr_confidence)}%",
                    center_style,
                ),
            ],
        ]

        summary_table = Table(
            summary_rows,
            colWidths=[
                32 * mm,
                28 * mm,
                25 * mm,
                30 * mm,
                25 * mm,
                25 * mm,
                30 * mm,
                32 * mm,
            ],
            repeatRows=1,
        )

        summary_table.setStyle(
            TableStyle(
                [
                    (
                        "BACKGROUND",
                        (0, 0),
                        (-1, 0),
                        colors.HexColor(
                            "#123B8A"
                        ),
                    ),
                    (
                        "TEXTCOLOR",
                        (0, 0),
                        (-1, 0),
                        colors.white,
                    ),
                    (
                        "GRID",
                        (0, 0),
                        (-1, -1),
                        0.4,
                        colors.HexColor(
                            "#AAAAAA"
                        ),
                    ),
                    (
                        "VALIGN",
                        (0, 0),
                        (-1, -1),
                        "MIDDLE",
                    ),
                    (
                        "ALIGN",
                        (0, 0),
                        (-1, -1),
                        "CENTER",
                    ),
                    (
                        "BACKGROUND",
                        (0, 1),
                        (-1, 1),
                        colors.HexColor(
                            "#F8F9FA"
                        ),
                    ),
                    (
                        "LEFTPADDING",
                        (0, 0),
                        (-1, -1),
                        4,
                    ),
                    (
                        "RIGHTPADDING",
                        (0, 0),
                        (-1, -1),
                        4,
                    ),
                    (
                        "TOPPADDING",
                        (0, 0),
                        (-1, -1),
                        5,
                    ),
                    (
                        "BOTTOMPADDING",
                        (0, 0),
                        (-1, -1),
                        5,
                    ),
                    (
                        "TEXTCOLOR",
                        (1, 1),
                        (1, 1),
                        status_color(
                            status
                        ),
                    ),
                ]
            )
        )

        story.append(
            summary_table
        )

        story.append(
            Spacer(
                1,
                8,
            )
        )

        # =================================================
        # LOGO + BARCODE
        # =================================================

        logo = sample.get(
            "logo",
            {},
        ) or {}

        barcode = sample.get(
            "barcode",
            {},
        ) or {}

        logo_status = safe_value(
            logo.get(
                "status"
            ),
            "NOT_CHECKED",
        ).upper()

        logo_score = logo.get(
            "score",
            None,
        )

        barcode_status = safe_value(
            barcode.get(
                "status"
            ),
            "NOT_CHECKED",
        ).upper()

        barcode_score = barcode.get(
            "score",
            None,
        )

        logo_score_text = (
            "-"
            if logo_score is None
            else f"{logo_score}%"
        )

        barcode_score_text = (
            "-"
            if barcode_score is None
            else f"{barcode_score}%"
        )

        verification_rows = [
            [
                Paragraph(
                    "<b>Check</b>",
                    normal_style,
                ),
                Paragraph(
                    "<b>Status</b>",
                    normal_style,
                ),
                Paragraph(
                    "<b>Score</b>",
                    normal_style,
                ),
            ],
            [
                Paragraph(
                    "Logo",
                    normal_style,
                ),
                Paragraph(
                    pdf_text(
                        logo_status
                    ),
                    center_style,
                ),
                Paragraph(
                    logo_score_text,
                    center_style,
                ),
            ],
            [
                Paragraph(
                    "Barcode",
                    normal_style,
                ),
                Paragraph(
                    pdf_text(
                        barcode_status
                    ),
                    center_style,
                ),
                Paragraph(
                    barcode_score_text,
                    center_style,
                ),
            ],
        ]

        verification_table = Table(
            verification_rows,
            colWidths=[
                55 * mm,
                45 * mm,
                40 * mm,
            ],
        )

        verification_table.setStyle(
            TableStyle(
                [
                    (
                        "BACKGROUND",
                        (0, 0),
                        (-1, 0),
                        colors.HexColor(
                            "#495057"
                        ),
                    ),
                    (
                        "TEXTCOLOR",
                        (0, 0),
                        (-1, 0),
                        colors.white,
                    ),
                    (
                        "GRID",
                        (0, 0),
                        (-1, -1),
                        0.4,
                        colors.HexColor(
                            "#AAAAAA"
                        ),
                    ),
                    (
                        "VALIGN",
                        (0, 0),
                        (-1, -1),
                        "MIDDLE",
                    ),
                    (
                        "TEXTCOLOR",
                        (1, 1),
                        (1, 2),
                        colors.HexColor(
                            "#212529"
                        ),
                    ),
                    (
                        "LEFTPADDING",
                        (0, 0),
                        (-1, -1),
                        5,
                    ),
                    (
                        "RIGHTPADDING",
                        (0, 0),
                        (-1, -1),
                        5,
                    ),
                    (
                        "TOPPADDING",
                        (0, 0),
                        (-1, -1),
                        4,
                    ),
                    (
                        "BOTTOMPADDING",
                        (0, 0),
                        (-1, -1),
                        4,
                    ),
                ]
            )
        )

        story.append(
            verification_table
        )

        story.append(
            Spacer(
                1,
                8,
            )
        )

        # =================================================
        # FIELD COMPARISON
        # =================================================

        story.append(
            Paragraph(
                "FIELD COMPARISON",
                heading_style,
            )
        )

        rows = [
            [
                Paragraph(
                    "<b>Field</b>",
                    small_style,
                ),
                Paragraph(
                    "<b>Approval Label</b>",
                    small_style,
                ),
                Paragraph(
                    "<b>Sample Label</b>",
                    small_style,
                ),
                Paragraph(
                    "<b>Result</b>",
                    small_style,
                ),
                Paragraph(
                    "<b>Score</b>",
                    small_style,
                ),
            ]
        ]

        field_rows = field_result.get(
            "rows",
            [],
        ) or []

        if field_rows:

            for row in field_rows:

                row = row or {}

                field = safe_value(
                    row.get(
                        "field"
                    ),
                    "-",
                )

                approval_value = safe_value(
                    row.get(
                        "approval"
                    ),
                    "-",
                )

                sample_value = safe_value(
                    row.get(
                        "sample"
                    ),
                    "-",
                )

                row_status = safe_value(
                    row.get(
                        "status"
                    ),
                    "NOT_CHECKED",
                ).upper()

                row_score = row.get(
                    "score",
                    None,
                )

                score_text = (
                    "-"
                    if row_score is None
                    else f"{row_score}%"
                )

                rows.append(
                    [
                        Paragraph(
                            pdf_text(
                                field
                            ),
                            small_style,
                        ),
                        Paragraph(
                            pdf_text(
                                approval_value
                            ),
                            small_style,
                        ),
                        Paragraph(
                            pdf_text(
                                sample_value
                            ),
                            small_style,
                        ),
                        Paragraph(
                            pdf_text(
                                row_status
                            ),
                            center_style,
                        ),
                        Paragraph(
                            score_text,
                            center_style,
                        ),
                    ]
                )

        else:

            rows.append(
                [
                    Paragraph(
                        "No comparable fields found.",
                        small_style,
                    ),
                    Paragraph(
                        "-",
                        small_style,
                    ),
                    Paragraph(
                        "-",
                        small_style,
                    ),
                    Paragraph(
                        "NOT_CHECKED",
                        center_style,
                    ),
                    Paragraph(
                        "-",
                        center_style,
                    ),
                ]
            )

        field_table = Table(
            rows,
            repeatRows=1,
            colWidths=[
                42 * mm,
                72 * mm,
                72 * mm,
                28 * mm,
                25 * mm,
            ],
        )

        field_style_commands = [
            (
                "BACKGROUND",
                (0, 0),
                (-1, 0),
                colors.HexColor(
                    "#123B8A"
                ),
            ),
            (
                "TEXTCOLOR",
                (0, 0),
                (-1, 0),
                colors.white,
            ),
            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.35,
                colors.HexColor(
                    "#AAAAAA"
                ),
            ),
            (
                "VALIGN",
                (0, 0),
                (-1, -1),
                "TOP",
            ),
            (
                "LEFTPADDING",
                (0, 0),
                (-1, -1),
                4,
            ),
            (
                "RIGHTPADDING",
                (0, 0),
                (-1, -1),
                4,
            ),
            (
                "TOPPADDING",
                (0, 0),
                (-1, -1),
                4,
            ),
            (
                "BOTTOMPADDING",
                (0, 0),
                (-1, -1),
                4,
            ),
        ]

        # -------------------------------------------------
        # Color each status cell.
        # -------------------------------------------------

        for index, row in enumerate(
            field_rows,
            start=1,
        ):

            if not row:
                continue

            row_status = safe_value(
                row.get(
                    "status"
                ),
                "NOT_CHECKED",
            ).upper()

            field_style_commands.append(
                (
                    "TEXTCOLOR",
                    (3, index),
                    (3, index),
                    status_color(
                        row_status
                    ),
                )
            )

            field_style_commands.append(
                (
                    "FONTNAME",
                    (3, index),
                    (3, index),
                    "Helvetica-Bold",
                )
            )

        field_table.setStyle(
            TableStyle(
                field_style_commands
            )
        )

        story.append(
            field_table
        )

        story.append(
            Spacer(
                1,
                8,
            )
        )

        # =================================================
        # MISMATCH DETAILS
        # =================================================

        mismatches = sample.get(
            "mismatches",
            field_result.get(
                "mismatches",
                [],
            ),
        ) or []

        if mismatches:

            story.append(
                Paragraph(
                    "DETECTED CHANGES",
                    heading_style,
                )
            )

            mismatch_rows = [
                [
                    Paragraph(
                        "<b>Field</b>",
                        small_style,
                    ),
                    Paragraph(
                        "<b>Status</b>",
                        small_style,
                    ),
                    Paragraph(
                        "<b>Approval</b>",
                        small_style,
                    ),
                    Paragraph(
                        "<b>Sample</b>",
                        small_style,
                    ),
                ]
            ]

            for mismatch in mismatches:

                mismatch = (
                    mismatch or {}
                )

                mismatch_field = (
                    safe_value(
                        mismatch.get(
                            "field"
                        ),
                        "-",
                    )
                )

                mismatch_status = (
                    safe_value(
                        mismatch.get(
                            "status"
                        ),
                        "MISMATCH",
                    ).upper()
                )
                mismatch_status = (
                    safe_value(
                        mismatch.get(
                            "status"
                        ),
                        "MISMATCH",
                    ).upper()
                )

                mismatch_approval = (
                    safe_value(
                        mismatch.get(
                            "approval"
                        ),
                        "-",
                    )
                )

                mismatch_sample = (
                    safe_value(
                        mismatch.get(
                            "sample"
                        ),
                        "-",
                    )
                )

                mismatch_rows.append(
                    [
                        Paragraph(
                            pdf_text(
                                mismatch_field
                            ),
                            small_style,
                        ),
                        Paragraph(
                            pdf_text(
                                mismatch_status
                            ),
                            center_style,
                        ),
                        Paragraph(
                            pdf_text(
                                mismatch_approval
                            ),
                            small_style,
                        ),
                        Paragraph(
                            pdf_text(
                                mismatch_sample
                            ),
                            small_style,
                        ),
                    ]
                )

            mismatch_table = Table(
                mismatch_rows,
                repeatRows=1,
                colWidths=[
                    40 * mm,
                    30 * mm,
                    85 * mm,
                    85 * mm,
                ],
            )

            mismatch_style = [
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.HexColor(
                        "#842029"
                    ),
                ),
                (
                    "TEXTCOLOR",
                    (0, 0),
                    (-1, 0),
                    colors.white,
                ),
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.35,
                    colors.HexColor(
                        "#AAAAAA"
                    ),
                ),
                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "TOP",
                ),
                (
                    "LEFTPADDING",
                    (0, 0),
                    (-1, -1),
                    4,
                ),
                (
                    "RIGHTPADDING",
                    (0, 0),
                    (-1, -1),
                    4,
                ),
                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    4,
                ),
                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    4,
                ),
            ]

            for index, mismatch in enumerate(
                mismatches,
                start=1,
            ):

                mismatch_status = safe_value(
                    mismatch.get(
                        "status"
                    ),
                    "MISMATCH",
                ).upper()

                mismatch_style.append(
                    (
                        "TEXTCOLOR",
                        (1, index),
                        (1, index),
                        status_color(
                            mismatch_status
                        ),
                    )
                )

                mismatch_style.append(
                    (
                        "FONTNAME",
                        (1, index),
                        (1, index),
                        "Helvetica-Bold",
                    )
                )

            mismatch_table.setStyle(
                TableStyle(
                    mismatch_style
                )
            )

            story.append(
                mismatch_table
            )

            story.append(
                Spacer(
                    1,
                    8,
                )
            )

        # =================================================
        # OCR TEXT
        # =================================================

        sample_ocr = safe_value(
            sample.get(
                "ocr_text"
            ),
            "",
        )

        if sample_ocr:

            story.append(
                Paragraph(
                    "SAMPLE OCR TEXT",
                    heading_style,
                )
            )

            story.append(
                Table(
                    [
                        [
                            Paragraph(
                                pdf_text(
                                    sample_ocr
                                ),
                                small_style,
                            )
                        ]
                    ],
                    colWidths=[
                        255 * mm
                    ],
                    style=TableStyle(
                        [
                            (
                                "BACKGROUND",
                                (0, 0),
                                (-1, -1),
                                colors.HexColor(
                                    "#F8F9FA"
                                ),
                            ),
                            (
                                "BOX",
                                (0, 0),
                                (-1, -1),
                                0.4,
                                colors.HexColor(
                                    "#CCCCCC"
                                ),
                            ),
                            (
                                "LEFTPADDING",
                                (0, 0),
                                (-1, -1),
                                6,
                            ),
                            (
                                "RIGHTPADDING",
                                (0, 0),
                                (-1, -1),
                                6,
                            ),
                            (
                                "TOPPADDING",
                                (0, 0),
                                (-1, -1),
                                6,
                            ),
                            (
                                "BOTTOMPADDING",
                                (0, 0),
                                (-1, -1),
                                6,
                            ),
                        ]
                    ),
                )
            )

        # =================================================
        # PAGE BREAK BETWEEN SAMPLES
        # =================================================

        if sample_index < len(
            samples
        ):

            story.append(
                PageBreak()
            )

    # =====================================================
    # FOOTER / FINAL BUILD
    # =====================================================

    def add_page_number(
        canvas,
        doc,
    ):
        """
        Add footer with page number and report title.
        """

        canvas.saveState()

        width, height = (
            landscape(A4)
        )

        canvas.setFont(
            "Helvetica",
            7,
        )

        canvas.setFillColor(
            colors.HexColor(
                "#666666"
            )
        )

        canvas.drawString(
            10 * mm,
            6 * mm,
            "Label QC Checker Pro",
        )

        canvas.drawRightString(
            width - 10 * mm,
            6 * mm,
            f"Page {doc.page}",
        )

        canvas.restoreState()

    # =====================================================
    # BUILD PDF
    # =====================================================

    doc.build(
        story,
        onFirstPage=add_page_number,
        onLaterPages=add_page_number,
    )

    # =====================================================
    # VERIFY OUTPUT
    # =====================================================

    if not pdf_path.exists():

        raise IOError(
            "PDF generation failed. "
            f"File was not created: {pdf_path}"
        )

    return str(
        pdf_path
    )