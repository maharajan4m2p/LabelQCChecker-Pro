# Label QC Checker Pro - Corrected Build

This build implements:

- Approval and Sample labels displayed side-by-side.
- Approval/master image always passed explicitly to results.
- Line-by-line OCR comparison in reading order.
- Word-level changes inside each line.
- Coordinate mapping from upscaled OCR images back to the original image.
- Exact line-restricted highlighting.
- RED = changed/modified
- ORANGE = missing from Sample (highlighted on Approval)
- BLUE = extra in Sample
- GREEN = matching text (legend/line table)
- Combined side-by-side highlighted image.
- Detailed line-by-line results table.
- PDF report endpoint remains available.

Important architecture:
OCR boxes are generated from Tesseract's processed image and converted back
to ORIGINAL IMAGE coordinates before being returned. The highlighter then
uses the line index plus normalized token matching, preventing a repeated
word elsewhere on the label from being highlighted.

Run:
    python app.py

Then open:
    http://127.0.0.1:5000
