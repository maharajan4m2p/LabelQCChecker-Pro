"""
=========================================================
Label QC Checker Pro
Flask Application
=========================================================

Main responsibilities:

    1. Upload Approval Label
    2. Upload one/multiple Sample Labels
    3. Accept images and PDF files
    4. Convert PDF files to images
    5. Run Label Comparison
    6. Display Results
    7. Serve uploaded/generated images
    8. Generate PDF report
=========================================================
"""


# =========================================================
# IMPORTS
# =========================================================

from pathlib import Path
import uuid
import logging
import json

import numpy as np
import fitz

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    send_from_directory,
    flash,
    jsonify,
)

from werkzeug.utils import secure_filename


# =========================================================
# PROJECT IMPORTS
# =========================================================

from config import (
    UPLOAD_DIR,
    OUTPUT_DIR,
    ALLOWED_EXTENSIONS,
    MAX_CONTENT_LENGTH,
)

from label_compare import compare_labels

from report_generator import (
    build_pdf_report
)


# =========================================================
# FLASK APPLICATION
# =========================================================

app = Flask(__name__)


# =========================================================
# SECRET KEY
# =========================================================

app.secret_key = (
    "label-qc-checker-secret-change-me"
)


# =========================================================
# APPLICATION CONFIGURATION
# =========================================================

app.config[
    "MAX_CONTENT_LENGTH"
] = MAX_CONTENT_LENGTH


# =========================================================
# FOLDERS
# =========================================================

UPLOAD_DIR = Path(
    UPLOAD_DIR
)

OUTPUT_DIR = Path(
    OUTPUT_DIR
)


# =========================================================
# CREATE REQUIRED DIRECTORIES
# =========================================================

UPLOAD_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =========================================================
# LOGGING
# =========================================================

logging.basicConfig(
    level=logging.INFO,
)

app.logger.setLevel(
    logging.INFO
)


# =========================================================
# ALLOWED FILE CHECK
# =========================================================

def allowed_file(filename):
    """
    Check whether an uploaded file has
    an allowed extension.

    Supported examples:

        label.jpg
        label.jpeg
        label.png
        label.webp
        label.pdf

    Returns:

        True  -> allowed
        False -> not allowed
    """

    # -----------------------------------------------------
    # Empty filename
    # -----------------------------------------------------

    if not filename:

        return False


    # -----------------------------------------------------
    # Convert to string
    # -----------------------------------------------------

    filename = str(
        filename
    ).strip()


    # -----------------------------------------------------
    # Filename must contain extension
    # -----------------------------------------------------

    if "." not in filename:

        return False


    # -----------------------------------------------------
    # Extract extension
    # -----------------------------------------------------

    extension = (
        filename
        .rsplit(
            ".",
            1
        )[1]
        .lower()
    )


    # -----------------------------------------------------
    # Normalize configured extensions
    # -----------------------------------------------------

    allowed = {
        str(ext)
        .lower()
        .lstrip(".")
        for ext in ALLOWED_EXTENSIONS
    }


    # -----------------------------------------------------
    # Check extension
    # -----------------------------------------------------

    return (
        extension in allowed
    )


# =========================================================
# CREATE UNIQUE FILENAME
# =========================================================

def create_unique_filename(filename):
    """
    Create a safe unique filename.

    Example:

        approval.pdf

    becomes:

        approval_a82f72c91d21.pdf
    """

    # -----------------------------------------------------
    # Secure original filename
    # -----------------------------------------------------

    safe_name = secure_filename(
        filename
    )


    # -----------------------------------------------------
    # Validate filename
    # -----------------------------------------------------

    if not safe_name:

        raise ValueError(
            "Invalid filename."
        )


    # -----------------------------------------------------
    # Extract path information
    # -----------------------------------------------------

    original = Path(
        safe_name
    )


    # -----------------------------------------------------
    # Generate unique ID
    # -----------------------------------------------------

    unique_id = (
        uuid.uuid4()
        .hex[:12]
    )


    # -----------------------------------------------------
    # Return unique filename
    # -----------------------------------------------------

    return (
        f"{original.stem}_"
        f"{unique_id}"
        f"{original.suffix.lower()}"
    )


# =========================================================
# SAVE UPLOAD
# =========================================================

def save_upload(
    file_storage,
    folder
):
    """
    Safely save an uploaded file.

    Returns:

        Full filesystem path as string.
    """

    # -----------------------------------------------------
    # Validate uploaded file object
    # -----------------------------------------------------

    if (
        file_storage is None
        or not file_storage.filename
    ):

        return None


    # -----------------------------------------------------
    # Get original filename
    # -----------------------------------------------------

    original_filename = (
        file_storage.filename
    )


    # -----------------------------------------------------
    # Validate extension
    # -----------------------------------------------------

    if not allowed_file(
        original_filename
    ):

        raise ValueError(
            "Unsupported file type: "
            f"{original_filename}"
        )


    # -----------------------------------------------------
    # Create unique safe filename
    # -----------------------------------------------------

    filename = (
        create_unique_filename(
            original_filename
        )
    )


    # -----------------------------------------------------
    # Create destination folder
    # -----------------------------------------------------

    destination_folder = Path(
        folder
    )

    destination_folder.mkdir(
        parents=True,
        exist_ok=True,
    )


    # -----------------------------------------------------
    # Create destination path
    # -----------------------------------------------------

    destination = (
        destination_folder
        / filename
    )


    # -----------------------------------------------------
    # Save uploaded file
    # -----------------------------------------------------

    file_storage.save(
        str(destination)
    )


    # -----------------------------------------------------
    # Verify saved file
    # -----------------------------------------------------

    if not destination.exists():

        raise IOError(
            "Unable to save uploaded file: "
            f"{destination}"
        )


    # -----------------------------------------------------
    # Return saved path
    # -----------------------------------------------------

    return str(
        destination
    )
    # =========================================================
# PDF TO IMAGE CONVERSION
# =========================================================

def convert_pdf_to_image(file_path):
    """
    Convert the first page of a PDF into a PNG image.

    Image files are returned unchanged.

    PDF files are converted to PNG because the
    comparison/OCR engine expects an image path.

    Returns:
        String path to the image.
    """

    # -----------------------------------------------------
    # Convert input to Path object
    # -----------------------------------------------------

    file_path = Path(
        file_path
    )


    # -----------------------------------------------------
    # If this is already an image, return unchanged
    # -----------------------------------------------------

    if file_path.suffix.lower() != ".pdf":

        return str(
            file_path
        )


    # -----------------------------------------------------
    # Check PDF exists
    # -----------------------------------------------------

    if not file_path.exists():

        raise FileNotFoundError(
            "PDF file not found: "
            f"{file_path}"
        )


    pdf_document = None


    try:

        # =================================================
        # OPEN PDF
        # =================================================

        app.logger.info(
            "Opening PDF: %s",
            file_path,
        )

        pdf_document = fitz.open(
            str(file_path)
        )


        # =================================================
        # CHECK PAGE COUNT
        # =================================================

        if pdf_document.page_count <= 0:

            raise ValueError(
                "PDF contains no pages."
            )


        # =================================================
        # USE FIRST PAGE
        # =================================================

        page = pdf_document.load_page(0)


        # =================================================
        # RENDER PDF PAGE
        # =================================================

        matrix = fitz.Matrix(
            2.5,
            2.5
        )


        pixmap = page.get_pixmap(
            matrix=matrix,
            alpha=False,
        )


        # =================================================
        # CREATE OUTPUT FILE NAME
        # =================================================

        output_filename = (
            f"{file_path.stem}"
            f"_pdf_page1.png"
        )


        output_path = (
            OUTPUT_DIR
            / output_filename
        )


        # =================================================
        # SAVE PNG
        # =================================================

        pixmap.save(
            str(output_path)
        )


        # =================================================
        # VERIFY OUTPUT
        # =================================================

        if not output_path.exists():

            raise IOError(
                "PDF was converted but "
                "PNG file was not created."
            )


        if output_path.stat().st_size <= 0:

            raise IOError(
                "Converted PNG file is empty."
            )


        app.logger.info(
            "PDF converted successfully: %s",
            output_path,
        )


        # =================================================
        # RETURN PNG PATH
        # =================================================

        return str(
            output_path
        )


    except Exception as exc:

        app.logger.exception(
            "PDF conversion failed: %s",
            file_path,
        )

        raise ValueError(
            "Unable to convert PDF to image: "
            f"{exc}"
        )


    finally:

        # =================================================
        # ALWAYS CLOSE PDF
        # =================================================

        if pdf_document is not None:

            try:

                pdf_document.close()

            except Exception:

                app.logger.warning(
                    "Unable to close PDF: %s",
                    file_path,
                )
# =========================================================
# CONVERT RESULT TO JSON-SAFE DATA
# =========================================================

# =========================================================
# CONVERT RESULT TO JSON-SAFE DATA
# =========================================================

def make_json_safe(value):
    """
    Recursively convert comparison results into
    standard Python JSON-compatible values.

    Handles:

        Path
        NumPy ndarray
        NumPy scalar
        dict
        list
        tuple
        set
        bytes
    """

    # -----------------------------------------------------
    # pathlib.Path
    # -----------------------------------------------------

    if isinstance(value, Path):

        return str(value)


    # -----------------------------------------------------
    # NumPy ndarray
    # -----------------------------------------------------

    if isinstance(value, np.ndarray):

        return make_json_safe(
            value.tolist()
        )


    # -----------------------------------------------------
    # NumPy scalar
    # -----------------------------------------------------

    if isinstance(value, np.generic):

        return make_json_safe(
            value.item()
        )


    # -----------------------------------------------------
    # Dictionary
    # -----------------------------------------------------

    if isinstance(value, dict):

        return {
            str(key): make_json_safe(val)
            for key, val in value.items()
        }


    # -----------------------------------------------------
    # List
    # -----------------------------------------------------

    if isinstance(value, list):

        return [
            make_json_safe(item)
            for item in value
        ]


    # -----------------------------------------------------
    # Tuple
    # -----------------------------------------------------

    if isinstance(value, tuple):

        return [
            make_json_safe(item)
            for item in value
        ]


    # -----------------------------------------------------
    # Set
    # -----------------------------------------------------

    if isinstance(value, set):

        return [
            make_json_safe(item)
            for item in value
        ]


    # -----------------------------------------------------
    # Bytes
    # -----------------------------------------------------

    if isinstance(value, bytes):

        return value.decode(
            "utf-8",
            errors="replace"
        )


    # -----------------------------------------------------
    # Normal Python value
    # -----------------------------------------------------

    return value


# =========================================================
# GET WEB FILE URL
# =========================================================

def get_file_url(file_path):
    """
    Convert a filesystem path into a Flask URL.

    Files inside UPLOAD_DIR are served through:
        /uploads/...

    Files inside OUTPUT_DIR are served through:
        /outputs/...

    Returns:
        Flask URL string or None.
    """

    # -----------------------------------------------------
    # Empty path
    # -----------------------------------------------------

    if not file_path:

        return None


    try:

        path = Path(
            file_path
        )


        # =================================================
        # CHECK UPLOAD DIRECTORY
        # =================================================

        try:

            relative_upload = (
                path.relative_to(
                    UPLOAD_DIR
                )
            )

            return url_for(
                "uploads",
                filename=relative_upload.as_posix(),
            )

        except ValueError:

            pass


        # =================================================
        # CHECK OUTPUT DIRECTORY
        # =================================================

        try:

            relative_output = (
                path.relative_to(
                    OUTPUT_DIR
                )
            )

            return url_for(
                "outputs",
                filename=relative_output.as_posix(),
            )

        except ValueError:

            pass


    except Exception:

        app.logger.exception(
            "Unable to create file URL."
        )


    return None


# =========================================================
# INDEX PAGE
# =========================================================

@app.route(
    "/",
    methods=["GET"],
)
def index():
    """
    Display the Label QC Checker upload page.
    """

    return render_template(
        "index.html"
    )
    # =========================================================
# COMPARE LABELS
# =========================================================

@app.route(
    "/compare",
    methods=["POST"],
)
def compare():
    """
    Receive:

        approval_label
        sample_labels[]

    Then:

        1. Validate files
        2. Save files
        3. Convert PDF files to PNG
        4. Run label comparison
        5. Create image URLs
        6. Render results.html
    """

    try:

        # =================================================
        # GET APPROVAL LABEL
        # =================================================

        approval = request.files.get(
            "approval_label"
        )


        # =================================================
        # CHECK APPROVAL LABEL
        # =================================================

        if (
            approval is None
            or not approval.filename
        ):

            flash(
                "Please select an approval label.",
                "error",
            )

            return redirect(
                url_for("index")
            )


        # =================================================
        # VALIDATE APPROVAL FILE
        # =================================================

        if not allowed_file(
            approval.filename
        ):

            flash(
                "Unsupported approval label file type.",
                "error",
            )

            return redirect(
                url_for("index")
            )


        # =================================================
        # GET SAMPLE LABELS
        # =================================================

        samples = request.files.getlist(
            "sample_labels"
        )


        # =================================================
        # REMOVE EMPTY SAMPLE FILES
        # =================================================

        samples = [
            file
            for file in samples
            if file is not None
            and file.filename
        ]


        # =================================================
        # CHECK SAMPLE LABELS
        # =================================================

        if not samples:

            flash(
                "Please select at least one sample label.",
                "error",
            )

            return redirect(
                url_for("index")
            )


        # =================================================
        # VALIDATE SAMPLE FILES
        # =================================================

        invalid_samples = []

        for sample in samples:

            if not allowed_file(
                sample.filename
            ):

                invalid_samples.append(
                    sample.filename
                )


        if invalid_samples:

            flash(
                "Unsupported sample file type: "
                + ", ".join(
                    invalid_samples
                ),
                "error",
            )

            return redirect(
                url_for("index")
            )


        # =================================================
        # SAVE APPROVAL LABEL
        # =================================================

        approval_path = save_upload(
            approval,
            UPLOAD_DIR,
        )


        if not approval_path:

            raise ValueError(
                "Approval label could not be saved."
            )


        app.logger.info(
            "Approval label saved: %s",
            approval_path,
        )


        # =================================================
        # SAVE SAMPLE LABELS
        # =================================================

        sample_paths = []


        for sample_file in samples:

            sample_path = save_upload(
                sample_file,
                UPLOAD_DIR,
            )


            if sample_path:

                sample_paths.append(
                    sample_path
                )


        # =================================================
        # CHECK SAVED SAMPLES
        # =================================================

        if not sample_paths:

            raise ValueError(
                "No valid sample labels were uploaded."
            )


        app.logger.info(
            "Sample labels saved: %s",
            sample_paths,
        )


        # =================================================
        # CONVERT APPROVAL PDF
        # =================================================

        app.logger.info(
            "Checking approval file format..."
        )


        approval_path = (
            convert_pdf_to_image(
                approval_path
            )
        )


        app.logger.info(
            "Approval file ready: %s",
            approval_path,
        )


        # =================================================
        # CONVERT SAMPLE PDF FILES
        # =================================================

        converted_sample_paths = []


        for sample_path in sample_paths:

            converted_path = (
                convert_pdf_to_image(
                    sample_path
                )
            )


            converted_sample_paths.append(
                converted_path
            )


        sample_paths = (
            converted_sample_paths
        )


        app.logger.info(
            "Sample files ready: %s",
            sample_paths,
        )


        # =================================================
        # RUN LABEL COMPARISON
        # =================================================

        app.logger.info(
            "Starting label comparison..."
        )


        result = compare_labels(
            approval_path,
            sample_paths,
        )

# =================================================
# CONVERT ALL RESULT DATA TO JSON-SAFE VALUES
# =================================================

        result = make_json_safe(
            result
        )

# =================================================
# FORCE JSON SERIALIZATION TEST
# =================================================

        try:

            result = json.loads(
                json.dumps(
                    result,
                    ensure_ascii=False,
                )
            )

        except TypeError as exc:

            app.logger.exception(
                "Result still contains a non-JSON value."
            )

            raise ValueError(
                "Comparison result contains "
                "non-JSON-serializable data: "
                f"{exc}"
            )


        # =================================================
        # CHECK COMPARISON RESULT
        # =================================================

        if result is None:

            raise ValueError(
                "Comparison engine returned no result."
            )


        app.logger.info(
            "Comparison completed successfully."
        )


        app.logger.info(
            "Result type: %s",
            type(result).__name__,
        )


        # =================================================
        # CREATE APPROVAL URL
        # =================================================

        approval_url = get_file_url(
            approval_path
        )


        # =================================================
        # CREATE SAMPLE URLS
        # =================================================

        sample_urls = []


        for sample_path in sample_paths:

            sample_url = get_file_url(
                sample_path
            )


            sample_urls.append(
                {
                    "path": sample_path,

                    "url": sample_url,

                    "filename": Path(
                        sample_path
                    ).name,
                }
            )

        # =================================================
        # FINAL JSON SAFETY CHECK
        # =================================================

        try:

            json.dumps(
                result,
                ensure_ascii=False,
            )

            app.logger.info(
                "Final comparison result is JSON serializable."
            )

        except TypeError as exc:

            app.logger.exception(
                "Final result is NOT JSON serializable."
            )

            raise ValueError(
                "Final comparison result is not JSON serializable: "
                f"{exc}"
            )

        # =================================================
        # RENDER RESULTS PAGE
        # =================================================

        return render_template(
            "results.html",

            # ---------------------------------------------
            # Main comparison result
            # ---------------------------------------------

            result=result,


            # ---------------------------------------------
            # Approval information
            # ---------------------------------------------

            approval_path=approval_path,

            approval_url=approval_url,


            # ---------------------------------------------
            # Sample information
            # ---------------------------------------------

            sample_paths=sample_paths,

            sample_urls=sample_urls,
        )


    # =====================================================
    # HANDLE COMPARISON ERRORS
    # =====================================================

    except Exception as exc:

        app.logger.exception(
            "Label comparison failed."
        )


        flash(
            f"Comparison failed: {exc}",
            "error",
        )


        return redirect(
            url_for("index")
        )
        # =========================================================
# PDF REPORT
# =========================================================

@app.route(
    "/report/pdf",
    methods=["POST"],
)
def report_pdf():
    """
    Create and return a PDF report.

    The frontend sends the comparison result
    as JSON to this endpoint.
    """

    try:

        # =================================================
        # GET JSON PAYLOAD
        # =================================================

        payload = request.get_json(
            force=True,
            silent=False,
        )


        # =================================================
        # CHECK PAYLOAD
        # =================================================

        if payload is None:

            return jsonify(
                {
                    "error": (
                        "No comparison result "
                        "was supplied."
                    )
                }
            ), 400


        # =================================================
        # BUILD PDF REPORT
        # =================================================

        pdf_path = build_pdf_report(
            payload,
            OUTPUT_DIR,
        )


        # =================================================
        # CHECK PDF PATH
        # =================================================

        if not pdf_path:

            return jsonify(
                {
                    "error": (
                        "PDF generator returned "
                        "an empty path."
                    )
                }
            ), 500


        # =================================================
        # CONVERT TO PATH OBJECT
        # =================================================

        pdf_path = Path(
            pdf_path
        )


        # =================================================
        # VERIFY PDF EXISTS
        # =================================================

        if not pdf_path.exists():

            return jsonify(
                {
                    "error": (
                        "Generated PDF does not exist: "
                        f"{pdf_path}"
                    )
                }
            ), 500


        # =================================================
        # SEND PDF TO USER
        # =================================================

        return send_from_directory(
            directory=str(
                pdf_path.parent
            ),

            path=pdf_path.name,

            as_attachment=True,

            download_name=pdf_path.name,
        )


    # =====================================================
    # HANDLE PDF REPORT ERROR
    # =====================================================

    except Exception as exc:

        app.logger.exception(
            "PDF report generation failed."
        )


        return jsonify(
            {
                "error": str(exc)
            }
        ), 400
        # =========================================================
# SERVE UPLOADED FILES
# =========================================================

@app.route(
    "/uploads/<path:filename>"
)
def uploads(filename):
    """
    Serve uploaded files.

    Used by results.html to display:
        - uploaded approval files
        - uploaded sample files
    """

    return send_from_directory(
        directory=str(
            UPLOAD_DIR
        ),
        path=filename,
    )


# =========================================================
# SERVE OUTPUT FILES
# =========================================================

@app.route(
    "/outputs/<path:filename>"
)
def outputs(filename):
    """
    Serve generated output files.

    Used by results.html to display:
        - converted PDF images
        - highlighted images
        - comparison images
        - generated reports
    """

    return send_from_directory(
        directory=str(
            OUTPUT_DIR
        ),
        path=filename,
    )
    # =========================================================
# HEALTH CHECK
# =========================================================

@app.route(
    "/health",
    methods=["GET"],
)
def health():
    """
    Simple health-check endpoint.

    Useful for:
        - Local testing
        - Docker
        - Render deployment
        - Monitoring
    """

    return jsonify(
        {
            "status": "ok",
            "service": "Label QC Checker Pro",
        }
    )


# =========================================================
# ERROR HANDLER - FILE TOO LARGE
# =========================================================

@app.errorhandler(413)
def request_entity_too_large(error):
    """
    Handle files exceeding MAX_CONTENT_LENGTH.
    """

    flash(
        "Uploaded file is too large.",
        "error",
    )

    return redirect(
        url_for("index")
    )


# =========================================================
# ERROR HANDLER - 404
# =========================================================

@app.errorhandler(404)
def page_not_found(error):
    """
    Handle missing pages/files.
    """

    # -----------------------------------------------------
    # API request
    # -----------------------------------------------------

    if request.path.startswith(
        "/api/"
    ):

        return jsonify(
            {
                "error": "Resource not found."
            }
        ), 404


    # -----------------------------------------------------
    # Normal browser request
    # -----------------------------------------------------

    return (
        render_template(
            "index.html"
        ),
        404,
    )


# =========================================================
# ERROR HANDLER - 500
# =========================================================

@app.errorhandler(500)
def internal_server_error(error):
    """
    Handle unexpected server errors.
    """

    app.logger.error(
        "Internal server error: %s",
        error,
    )


    # -----------------------------------------------------
    # API request
    # -----------------------------------------------------

    if request.path.startswith(
        "/api/"
    ):

        return jsonify(
            {
                "error": (
                    "Internal server error."
                )
            }
        ), 500


    # -----------------------------------------------------
    # Normal browser request
    # -----------------------------------------------------

    flash(
        "An internal server error occurred.",
        "error",
    )


    return redirect(
        url_for("index")
    )


# =========================================================
# APPLICATION START
# =========================================================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True,
    )