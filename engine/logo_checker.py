"""
=========================================================
Label QC Checker Pro
Logo Checker
=========================================================

Purpose:
    Compare logo regions between Approval and Sample labels.

Important:
    This checker is intentionally conservative.

Statuses:
    PASS
    FAIL
    NOT_FOUND
    NOT_CHECKED

Features:
    - Safe image loading
    - Automatic logo-region crop
    - Edge/content detection
    - Grayscale structural comparison
    - Binary shape comparison
    - Histogram comparison
    - SSIM-like structural score
    - Multi-score logo verification
    - Avoids background-only PASS
    - Handles different image sizes
=========================================================
"""

from pathlib import Path

import cv2
import numpy as np


class LogoChecker:
    """
    Conservative logo comparison engine.

    The checker should NOT say PASS simply because two
    rectangular regions have similar brightness/background.

    A PASS requires enough visual evidence and sufficient
    structural similarity.
    """

    # =====================================================
    # CONFIGURATION
    # =====================================================

    # Minimum edge ratio required to consider a crop
    # visually meaningful.
    MIN_EDGE_RATIO = 0.015

    # Minimum standard deviation required to avoid
    # treating a completely uniform background as a logo.
    MIN_STD = 8.0

    # Final PASS threshold.
    FINAL_PASS_THRESHOLD = 0.72

    # Strong structural threshold.
    STRUCTURE_PASS_THRESHOLD = 0.68

    # Minimum content threshold for both images.
    CONTENT_THRESHOLD = 0.015

    # =====================================================
    # INITIALIZATION
    # =====================================================

    def __init__(
        self,
        pass_threshold=None,
    ):
        """
        Initialize LogoChecker.
        """

        if pass_threshold is None:

            self.pass_threshold = (
                self.FINAL_PASS_THRESHOLD
            )

        else:

            self.pass_threshold = float(
                pass_threshold
            )

    # =====================================================
    # LOAD IMAGE
    # =====================================================

    def load_image(
        self,
        path,
    ):
        """
        Safely load an image.
        """

        path = Path(
            path
        )

        if not path.exists():
            return None

        image = cv2.imread(
            str(path)
        )

        return image

    # =====================================================
    # CROP LOGO REGION
    # =====================================================

    def crop_logo_region(
        self,
        image,
    ):
        """
        Extract the default logo region.

        Current region:

            Top 2%  -> 30%
            Left 2% -> 45%

        This is intentionally conservative.

        If your actual logo is in another location, this
        function is the exact location to modify later.
        """

        if image is None:
            return None

        height, width = (
            image.shape[:2]
        )

        if (
            height <= 0
            or width <= 0
        ):
            return None

        y1 = int(
            height * 0.02
        )

        y2 = int(
            height * 0.30
        )

        x1 = int(
            width * 0.02
        )

        x2 = int(
            width * 0.45
        )

        # Safety boundaries.
        y1 = max(
            0,
            min(
                y1,
                height - 1,
            ),
        )

        y2 = max(
            y1 + 1,
            min(
                y2,
                height,
            ),
        )

        x1 = max(
            0,
            min(
                x1,
                width - 1,
            ),
        )

        x2 = max(
            x1 + 1,
            min(
                x2,
                width,
            ),
        )

        crop = image[
            y1:y2,
            x1:x2,
        ]

        if crop.size == 0:
            return None

        return crop

    # =====================================================
    # CREATE MULTIPLE LOGO REGIONS
    # =====================================================

    def get_logo_regions(
        self,
        image,
    ):
        """
        Generate several possible logo crops.

        This helps when the logo isn't exactly in one fixed
        position.

        The regions are intentionally limited to the upper
        part of the label.
        """

        if image is None:
            return []

        height, width = (
            image.shape[:2]
        )

        if (
            height < 20
            or width < 20
        ):
            return []

        regions = []

        # -------------------------------------------------
        # Region 1 - Original configured region
        # -------------------------------------------------

        region1 = image[
            int(height * 0.02):
            int(height * 0.30),

            int(width * 0.02):
            int(width * 0.45),
        ]

        if region1.size > 0:
            regions.append(
                region1
            )

        # -------------------------------------------------
        # Region 2 - Wider top-left
        # -------------------------------------------------

        region2 = image[
            0:
            int(height * 0.35),

            0:
            int(width * 0.60),
        ]

        if region2.size > 0:
            regions.append(
                region2
            )

        # -------------------------------------------------
        # Region 3 - Center-top
        # -------------------------------------------------

        region3 = image[
            0:
            int(height * 0.30),

            int(width * 0.15):
            int(width * 0.70),
        ]

        if region3.size > 0:
            regions.append(
                region3
            )

        return regions

    # =====================================================
    # VISUAL CONTENT
    # =====================================================

    def get_edge_ratio(
        self,
        crop,
    ):
        """
        Calculate edge density.
        """

        if (
            crop is None
            or crop.size == 0
        ):
            return 0.0

        gray = cv2.cvtColor(
            crop,
            cv2.COLOR_BGR2GRAY,
        )

        edges = cv2.Canny(
            gray,
            50,
            150,
        )

        return float(
            np.count_nonzero(
                edges
            )
        ) / float(
            edges.size
        )

    # =====================================================
    # VISUAL CONTENT CHECK
    # =====================================================

    def has_visual_content(
        self,
        crop,
    ):
        """
        Determine whether a crop contains enough visual
        structure to plausibly contain a logo.
        """

        if (
            crop is None
            or crop.size == 0
        ):
            return False

        gray = cv2.cvtColor(
            crop,
            cv2.COLOR_BGR2GRAY,
        )

        # -------------------------------------------------
        # Standard deviation
        # -------------------------------------------------

        standard_deviation = float(
            np.std(gray)
        )

        if (
            standard_deviation
            < self.MIN_STD
        ):
            return False

        # -------------------------------------------------
        # Edge density
        # -------------------------------------------------

        edge_ratio = (
            self.get_edge_ratio(
                crop
            )
        )

        if (
            edge_ratio
            < self.MIN_EDGE_RATIO
        ):
            return False

        return True

    # =====================================================
    # PREPROCESS LOGO
    # =====================================================

    def preprocess(
        self,
        image,
    ):
        """
        Prepare logo crop for comparison.
        """

        if (
            image is None
            or image.size == 0
        ):
            return None

        # Standard size.
        image = cv2.resize(
            image,
            (300, 180),
            interpolation=cv2.INTER_AREA,
        )

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY,
        )

        # Reduce small printing noise.
        gray = cv2.GaussianBlur(
            gray,
            (3, 3),
            0,
        )

        # Improve local contrast.
        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8),
        )

        gray = clahe.apply(
            gray
        )

        return gray

    # =====================================================
    # EDGE IMAGE
    # =====================================================

    def edge_image(
        self,
        gray,
    ):
        """
        Create normalized edge representation.
        """

        if (
            gray is None
            or gray.size == 0
        ):
            return None

        edges = cv2.Canny(
            gray,
            50,
            150,
        )

        # Slight dilation makes logos more robust against
        # small print/scan differences.
        kernel = np.ones(
            (2, 2),
            np.uint8,
        )

        edges = cv2.dilate(
            edges,
            kernel,
            iterations=1,
        )

        return edges

    # =====================================================
    # HISTOGRAM SIMILARITY
    # =====================================================

    def histogram_similarity(
        self,
        first,
        second,
    ):
        """
        Compare grayscale histograms.

        Returns:
            0..1
        """

        if (
            first is None
            or second is None
        ):
            return 0.0

        hist1 = cv2.calcHist(
            [first],
            [0],
            None,
            [64],
            [0, 256],
        )

        hist2 = cv2.calcHist(
            [second],
            [0],
            None,
            [64],
            [0, 256],
        )

        cv2.normalize(
            hist1,
            hist1,
            alpha=0,
            beta=1,
            norm_type=cv2.NORM_MINMAX,
        )

        cv2.normalize(
            hist2,
            hist2,
            alpha=0,
            beta=1,
            norm_type=cv2.NORM_MINMAX,
        )

        correlation = cv2.compareHist(
            hist1,
            hist2,
            cv2.HISTCMP_CORREL,
        )

        # Correlation is roughly -1..1.
        score = (
            correlation + 1.0
        ) / 2.0

        return max(
            0.0,
            min(
                1.0,
                float(score),
            ),
        )

    # =====================================================
    # PIXEL SIMILARITY
    # =====================================================

    def pixel_similarity(
        self,
        first,
        second,
    ):
        """
        Compare grayscale pixel values.
        """

        if (
            first is None
            or second is None
        ):
            return 0.0

        if first.shape != second.shape:

            second = cv2.resize(
                second,
                (
                    first.shape[1],
                    first.shape[0],
                ),
                interpolation=cv2.INTER_AREA,
            )

        difference = cv2.absdiff(
            first,
            second,
        )

        mean_difference = float(
            np.mean(
                difference
            )
        )

        similarity = (
            1.0
            -
            (
                mean_difference
                / 255.0
            )
        )

        return max(
            0.0,
            min(
                1.0,
                similarity,
            ),
        )

    # =====================================================
    # STRUCTURAL SIMILARITY
    # =====================================================

    def structural_similarity(
        self,
        first,
        second,
    ):
        """
        Compare structural edge information.

        This is more useful for logos than raw pixel
        similarity because logos may have slight changes
        in brightness/contrast.
        """

        first_edges = (
            self.edge_image(
                first
            )
        )

        second_edges = (
            self.edge_image(
                second
            )
        )

        if (
            first_edges is None
            or second_edges is None
        ):
            return 0.0

        if (
            first_edges.shape
            != second_edges.shape
        ):

            second_edges = cv2.resize(
                second_edges,
                (
                    first_edges.shape[1],
                    first_edges.shape[0],
                ),
                interpolation=cv2.INTER_NEAREST,
            )

        # -------------------------------------------------
        # Intersection over union for edges.
        # -------------------------------------------------

        first_binary = (
            first_edges > 0
        )

        second_binary = (
            second_edges > 0
        )

        intersection = np.logical_and(
            first_binary,
            second_binary,
        )

        union = np.logical_or(
            first_binary,
            second_binary,
        )

        union_count = int(
            np.count_nonzero(
                union
            )
        )

        if union_count == 0:
            return 0.0

        iou = float(
            np.count_nonzero(
                intersection
            )
        ) / float(
            union_count
        )

        return max(
            0.0,
            min(
                1.0,
                iou,
            ),
        )

    # =====================================================
    # COMBINED IMAGE COMPARISON
    # =====================================================

    def compare_images(
        self,
        first,
        second,
    ):
        """
        Compare two logo crops using multiple signals.

        Score components:

            Pixel similarity
            Histogram similarity
            Structural similarity

        Structural similarity receives the strongest weight.
        """

        first_processed = (
            self.preprocess(
                first
            )
        )

        second_processed = (
            self.preprocess(
                second
            )
        )

        if (
            first_processed is None
            or second_processed is None
        ):
            return 0.0

        pixel_score = (
            self.pixel_similarity(
                first_processed,
                second_processed,
            )
        )

        histogram_score = (
            self.histogram_similarity(
                first_processed,
                second_processed,
            )
        )

        structural_score = (
            self.structural_similarity(
                first_processed,
                second_processed,
            )
        )

        # -------------------------------------------------
        # Weighted score.
        #
        # Structure is most important for logos.
        # -------------------------------------------------

        score = (
            pixel_score * 0.20
            +
            histogram_score * 0.20
            +
            structural_score * 0.60
        )

        return max(
            0.0,
            min(
                1.0,
                float(score),
            ),
        )

    # =====================================================
    # BEST REGION
    # =====================================================

    def find_best_matching_region(
        self,
        approval_regions,
        sample_regions,
    ):
        """
        Compare possible logo regions and return the highest
        structural match.

        This makes the checker less dependent on a single
        hard-coded crop.
        """

        best_score = 0.0

        best_pair = None

        for approval_region in (
            approval_regions
        ):

            if not self.has_visual_content(
                approval_region
            ):
                continue

            for sample_region in (
                sample_regions
            ):

                if not self.has_visual_content(
                    sample_region
                ):
                    continue

                score = (
                    self.compare_images(
                        approval_region,
                        sample_region,
                    )
                )

                if score > best_score:

                    best_score = (
                        score
                    )

                    best_pair = (
                        approval_region,
                        sample_region,
                    )

        return (
            best_score,
            best_pair,
        )

    # =====================================================
    # MAIN COMPARE
    # =====================================================

    def compare(
        self,
        approval_path,
        sample_path,
    ):
        """
        Compare the logo regions of Approval and Sample.

        Returns:

            {
                "status": "PASS",
                "score": 92.4
            }
        """

        # -------------------------------------------------
        # Load images
        # -------------------------------------------------

        approval = self.load_image(
            approval_path
        )

        sample = self.load_image(
            sample_path
        )

        if (
            approval is None
            or sample is None
        ):

            return {
                "status": "NOT_FOUND",
                "score": 0.0,
                "message": (
                    "Unable to load one or both images."
                ),
            }

        # -------------------------------------------------
        # Get possible logo regions.
        # -------------------------------------------------

        approval_regions = (
            self.get_logo_regions(
                approval
            )
        )

        sample_regions = (
            self.get_logo_regions(
                sample
            )
        )

        if not approval_regions:

            return {
                "status": "NOT_FOUND",
                "score": 0.0,
                "message": (
                    "No usable logo region found "
                    "on approval label."
                ),
            }

        if not sample_regions:

            return {
                "status": "NOT_FOUND",
                "score": 0.0,
                "message": (
                    "No usable logo region found "
                    "on sample label."
                ),
            }

        # -------------------------------------------------
        # Check whether approval has actual visual content.
        # -------------------------------------------------

        approval_has_content = any(
            self.has_visual_content(
                region
            )
            for region
            in approval_regions
        )

        if not approval_has_content:

            return {
                "status": "NOT_FOUND",
                "score": 0.0,
                "message": (
                    "No reliable logo content detected "
                    "on approval label."
                ),
            }

        # -------------------------------------------------
        # Check whether sample has actual visual content.
        # -------------------------------------------------

        sample_has_content = any(
            self.has_visual_content(
                region
            )
            for region
            in sample_regions
        )

        if not sample_has_content:

            return {
                "status": "FAIL",
                "score": 0.0,
                "message": (
                    "Logo appears to be missing "
                    "from sample label."
                ),
            }

        # -------------------------------------------------
        # Compare best regions.
        # -------------------------------------------------

        score, best_pair = (
            self.find_best_matching_region(
                approval_regions,
                sample_regions,
            )
        )

        # -------------------------------------------------
        # No meaningful comparison.
        # -------------------------------------------------

        if best_pair is None:

            return {
                "status": "NOT_FOUND",
                "score": 0.0,
                "message": (
                    "Logo content could not be "
                    "reliably compared."
                ),
            }

        # -------------------------------------------------
        # Final PASS / FAIL.
        # -------------------------------------------------

        if (
            score
            >= self.pass_threshold
        ):

            status = "PASS"

        else:

            status = "FAIL"

        return {
            "status": status,
            "score": round(
                score * 100,
                1,
            ),
            "message": (
                "Logo regions compared successfully."
            ),
        }