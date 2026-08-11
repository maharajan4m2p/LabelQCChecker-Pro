"""
=========================================================
Label QC Checker Pro
Field Detection & Extraction Engine
=========================================================

Purpose:
    Extract structured business fields from OCR text.

Improved features:
    - Robust OCR field detection
    - FIELD_ALIASES support
    - Handles :, =, -, | and OCR punctuation
    - Prevents BUYER matching BUYER CODE incorrectly
    - Detects fields across multiple OCR lines
    - Supports multi-line field values
    - Handles OCR spaces and punctuation
    - Removes duplicate field labels from values
    - Cleans common OCR artifacts
    - Better extraction for garment/carton labels

=========================================================
"""

import re

from config import FIELD_ALIASES


class FieldDetector:
    """
    Detect important business fields from OCR text.
    """

    # =====================================================
    # IMPORTANT BUSINESS FIELDS
    # =====================================================

    FIELD_NAMES = [
        "BUYER",
        "BUYER CODE",
        "VENDOR",
        "PO NO",
        "STYLE NO",
        "DESCRIPTION",
        "COLOR",
        "SIZE",
        "QTY",
        "TOTAL QTY",
        "G.W.",
        "N.W.",
        "MEASUREMENT",
        "VOLUME",
        "CARTON NO",
        "COUNTRY OF ORIGIN",
        "DESTINATION",
        "PORT OF LOADING",
        "PORT OF DISCHARGE",
        "SHIPMENT MODE",
        "ETD",
        "ETA",
    ]

    # =====================================================
    # INITIALIZE
    # =====================================================

    def __init__(self):
        """
        Prepare normalized field/alias patterns.

        Longest aliases are checked first.

        This is important because:

            BUYER CODE

        must be checked before:

            BUYER
        """

        self.field_names = sorted(
            self.FIELD_NAMES,
            key=len,
            reverse=True,
        )

        self.alias_lookup = {}

        for field_name, aliases in FIELD_ALIASES.items():

            if aliases is None:
                aliases = []

            if isinstance(aliases, str):
                aliases = [aliases]

            cleaned_aliases = []

            # Always include the canonical field name.
            all_aliases = [
                field_name,
                *aliases,
            ]

            for alias in all_aliases:

                if not alias:
                    continue

                alias = self.normalize_label(
                    alias
                )

                if alias and alias not in cleaned_aliases:
                    cleaned_aliases.append(alias)

            # Longest first.
            cleaned_aliases.sort(
                key=len,
                reverse=True,
            )

            self.alias_lookup[field_name] = (
                cleaned_aliases
            )

    # =====================================================
    # NORMALIZE LABEL
    # =====================================================

    def normalize_label(self, value):
        """
        Normalize a field label.

        Examples:

            BUYER:
                BUYER

            BUYER CODE:
                BUYER CODE

            PO NO.:
                PO NO

            STYLE-NO:
                STYLE NO
        """

        if value is None:
            return ""

        value = str(value).upper().strip()

        # OCR punctuation normalization
        value = (
            value
            .replace("—", "-")
            .replace("–", "-")
            .replace("_", " ")
            .replace("|", " ")
        )

        # Remove punctuation around labels.
        value = re.sub(
            r"[.:;]+$",
            "",
            value,
        )

        # Convert separators to spaces.
        value = re.sub(
            r"[-]+",
            " ",
            value,
        )

        # Normalize spaces.
        value = re.sub(
            r"\s+",
            " ",
            value,
        )

        return value.strip()

    # =====================================================
    # NORMALIZE OCR TEXT
    # =====================================================

    def normalize_text(self, value):
        """
        Normalize OCR text without destroying useful
        characters such as /, -, ., %, parentheses, etc.
        """

        if value is None:
            return ""

        value = str(value)

        # Unicode dash normalization.
        value = (
            value
            .replace("—", "-")
            .replace("–", "-")
            .replace("−", "-")
        )

        # Common OCR separators.
        value = value.replace("\t", " ")

        # Collapse spaces.
        value = re.sub(
            r"[ \t]+",
            " ",
            value,
        )

        # Normalize excessive blank lines.
        value = re.sub(
            r"\n{2,}",
            "\n",
            value,
        )

        return value.strip()

    # =====================================================
    # CLEAN FIELD VALUE
    # =====================================================

    def clean_value(self, value):
        """
        Clean an extracted field value.

        Does NOT aggressively remove characters because
        garment labels commonly contain:

            /
            -
            .
            ,
            %
            x
            ()
        """

        if value is None:
            return ""

        value = self.normalize_text(value)

        # Remove leading separators.
        value = re.sub(
            r"^[\s:=;|,\-]+",
            "",
            value,
        )

        # Remove trailing separators.
        value = re.sub(
            r"[\s:=;|]+$",
            "",
            value,
        )

        # Remove accidental repeated spaces.
        value = re.sub(
            r"\s+",
            " ",
            value,
        )

        return value.strip()

    # =====================================================
    # FIELD START DETECTION
    # =====================================================

    def is_field_start(self, text):
        """
        Check whether an OCR line starts with a known field.

        Example:

            BUYER: ABC FASHION LTD.
                -> True

            BUYER CODE: BUY-1001
                -> True

            RANDOM TEXT
                -> False
        """

        if not text:
            return False

        normalized = self.normalize_label(
            text
        )

        # Longest labels first.
        for field in self.field_names:

            field_normalized = (
                self.normalize_label(field)
            )

            # Exact field.
            if normalized == field_normalized:
                return True

            # Field followed by separator/value.
            pattern = (
                r"^"
                + re.escape(field_normalized)
                + r"(?=\s|[:=|\-]|$)"
            )

            if re.match(
                pattern,
                normalized,
                flags=re.IGNORECASE,
            ):
                return True

        # Also check configured aliases.
        for aliases in self.alias_lookup.values():

            for alias in aliases:

                pattern = (
                    r"^"
                    + re.escape(alias)
                    + r"(?=\s|[:=|\-]|$)"
                )

                if re.match(
                    pattern,
                    normalized,
                    flags=re.IGNORECASE,
                ):
                    return True

        return False

    # =====================================================
    # FIND MATCHING FIELD
    # =====================================================

    def _match_field_start(self, text):
        """
        Return:

            (field_name, remaining_value)

        for a line such as:

            BUYER: ABC FASHION LTD.

        Result:

            (
                "BUYER",
                "ABC FASHION LTD."
            )
        """

        if not text:
            return None, ""

        original = self.normalize_text(
            text
        )

        if not original:
            return None, ""

        # -------------------------------------------------
        # Check every configured field.
        # -------------------------------------------------

        candidates = []

        for field_name, aliases in (
            self.alias_lookup.items()
        ):

            for alias in aliases:

                if not alias:
                    continue

                candidates.append(
                    (
                        len(alias),
                        field_name,
                        alias,
                    )
                )

        # Longest alias first.
        candidates.sort(
            key=lambda item: item[0],
            reverse=True,
        )

        for _, field_name, alias in candidates:

            # ---------------------------------------------
            # Exact label
            # ---------------------------------------------

            if original.upper() == alias.upper():

                return field_name, ""

            # ---------------------------------------------
            # Label + separator/value
            # ---------------------------------------------

            pattern = re.compile(
                r"^"
                + re.escape(alias)
                + r"(?=\s|[:=|;\-]|$)"
                ,
                flags=re.IGNORECASE,
            )

            match = pattern.match(
                original
            )

            if not match:
                continue

            remainder = original[
                match.end():
            ]

            # Remove separators after field.
            remainder = remainder.lstrip(
                " \t:=;|-"
            )

            remainder = self.clean_value(
                remainder
            )

            return field_name, remainder

        return None, ""

    # =====================================================
    # BUILD ALIAS PATTERN
    # =====================================================

    def build_alias_pattern(self, aliases):
        """
        Build a safe regex for configured aliases.
        """

        if not aliases:
            return r"(?!x)x"

        cleaned = []

        for alias in aliases:

            if not alias:
                continue

            alias = self.normalize_label(
                alias
            )

            if alias:
                cleaned.append(
                    re.escape(alias)
                )

        # Longest first to avoid:
        #
        # BUYER
        # matching before
        # BUYER CODE
        #

        cleaned.sort(
            key=len,
            reverse=True,
        )

        if not cleaned:
            return r"(?!x)x"

        return (
            r"(?:"
            + "|".join(cleaned)
            + r")"
        )

    # =====================================================
    # FIND FIELD VALUE
    # =====================================================

    def find_field_value(
        self,
        text,
        aliases,
    ):
        """
        Find a field value after its label.

        This method uses two strategies:

            1. Structured line-by-line extraction
            2. Regex fallback for OCR text blocks
        """

        if not text:
            return ""

        if aliases is None:
            aliases = []

        if isinstance(aliases, str):
            aliases = [aliases]

        # -------------------------------------------------
        # Strategy 1:
        # Line-by-line
        # -------------------------------------------------

        lines = str(text).splitlines()

        normalized_aliases = []

        for alias in aliases:

            normalized_alias = (
                self.normalize_label(alias)
            )

            if normalized_alias:
                normalized_aliases.append(
                    normalized_alias
                )

        # Longest first.
        normalized_aliases.sort(
            key=len,
            reverse=True,
        )

        for line in lines:

            line = self.normalize_text(
                line
            )

            if not line:
                continue

            upper_line = line.upper()

            for alias in normalized_aliases:

                # Exact label.
                if upper_line == alias.upper():
                    continue

                pattern = re.compile(
                    r"^"
                    + re.escape(alias)
                    + r"(?=\s|[:=|;\-]|$)"
                    ,
                    flags=re.IGNORECASE,
                )

                match = pattern.match(
                    line
                )

                if not match:
                    continue

                remainder = line[
                    match.end():
                ]

                remainder = remainder.lstrip(
                    " \t:=;|-"
                )

                remainder = self.clean_value(
                    remainder
                )

                if remainder:
                    return remainder

        # -------------------------------------------------
        # Strategy 2:
        # Regex fallback
        # -------------------------------------------------

        alias_pattern = (
            self.build_alias_pattern(
                normalized_aliases
            )
        )

        if not alias_pattern:
            return ""

        # Look for:
        #
        # FIELD : VALUE
        # FIELD = VALUE
        # FIELD - VALUE
        #

        pattern = re.compile(
            rf"(?im)"
            rf"^\s*"
            rf"{alias_pattern}"
            rf"\s*"
            rf"[:=|;\-]?"
            rf"\s*"
            rf"(.+?)"
            rf"\s*$"
        )

        match = pattern.search(
            str(text)
        )

        if match:

            value = self.clean_value(
                match.group(1)
            )

            if value:
                return value

        return ""

    # =====================================================
    # EXTRACT LINE STRUCTURE
    # =====================================================

    def _parse_lines(self, text):
        """
        Parse OCR text into field/value pairs.

        Supports multiline values.

        Example:

            DESCRIPTION:
            MEN'S T-SHIRT
            THIS SIDE UP KEEP DRY

        becomes:

            DESCRIPTION =
            MEN'S T-SHIRT THIS SIDE UP KEEP DRY
        """

        fields = {}

        if not text:
            return fields

        lines = [
            self.normalize_text(line)
            for line in str(text).splitlines()
        ]

        lines = [
            line
            for line in lines
            if line
        ]

        current_field = None
        current_values = []

        def save_current():

            nonlocal current_field
            nonlocal current_values

            if current_field is None:
                return

            value = " ".join(
                current_values
            )

            value = self.clean_value(
                value
            )

            if value:
                fields[current_field] = value

            current_field = None
            current_values = []

        for line in lines:

            field_name, value = (
                self._match_field_start(
                    line
                )
            )

            # -------------------------------------------------
            # New field found
            # -------------------------------------------------

            if field_name:

                save_current()

                current_field = field_name

                if value:
                    current_values = [
                        value
                    ]

                continue

            # -------------------------------------------------
            # Continuation of previous field
            # -------------------------------------------------

            if current_field:

                # Ignore obvious OCR noise.
                if self._is_noise_line(line):
                    continue

                current_values.append(
                    line
                )

        # Save final field.
        save_current()

        return fields

    # =====================================================
    # OCR NOISE DETECTION
    # =====================================================

    def _is_noise_line(self, line):
        """
        Detect obvious OCR noise.

        Prevents random symbols from becoming part of
        a field value.
        """

        if not line:
            return True

        cleaned = re.sub(
            r"[^A-Z0-9]+",
            "",
            line.upper(),
        )

        if not cleaned:
            return True

        # Single-character punctuation noise.
        if len(cleaned) <= 1:
            return True

        return False

    # =====================================================
    # REMOVE DUPLICATE FIELD PREFIX
    # =====================================================

    def _remove_field_prefix(
        self,
        field_name,
        value,
    ):
        """
        Remove accidental repeated field names.

        Example:

            DESCRIPTION MEN'S T-SHIRT DESCRIPTION COLOR BLACK

        should not leave the repeated label inside the value.
        """

        if not value:
            return ""

        aliases = self.alias_lookup.get(
            field_name,
            [],
        )

        cleaned = self.clean_value(
            value
        )

        changed = True

        while changed:

            changed = False

            for alias in aliases:

                pattern = re.compile(
                    r"^"
                    + re.escape(alias)
                    + r"(?=\s|[:=|;\-]|$)"
                    ,
                    flags=re.IGNORECASE,
                )

                match = pattern.match(
                    cleaned
                )

                if match:

                    cleaned = (
                        cleaned[
                            match.end():
                        ]
                        .lstrip(
                            " \t:=;|-"
                        )
                    )

                    changed = True
                    break

        return self.clean_value(
            cleaned
        )

    # =====================================================
    # EXTRACT ALL FIELDS
    # =====================================================

    def extract_fields(self, text):
        """
        Extract all configured fields from OCR text.

        Primary extraction:
            Structured line parser

        Secondary extraction:
            Individual alias search

        Returns:
            {
                "BUYER": "...",
                "VENDOR": "...",
                "PO NO": "...",
                ...
            }
        """

        fields = {}

        # Initialize all configured fields.
        for field_name in FIELD_ALIASES.keys():
            fields[field_name] = ""

        # -------------------------------------------------
        # First pass:
        # Structured OCR parsing
        # -------------------------------------------------
        
        parsed_fields = self._parse_lines(
            text
        )

        for field_name, value in (
            parsed_fields.items()
        ):

            if field_name not in fields:
                fields[field_name] = ""

            fields[field_name] = (
                self._remove_field_prefix(
                    field_name,
                    value,
                )
            )

        # -------------------------------------------------
        # Second pass:
        # Individual field search
        #
        # This recovers fields missed by the line parser.
        # -------------------------------------------------

        for field_name, aliases in (
            FIELD_ALIASES.items()
        ):

            current_value = fields.get(
                field_name,
                "",
            )

            # Don't overwrite a good value.
            if current_value:
                continue

            value = self.find_field_value(
                text,
                aliases,
            )

            if value:

                value = (
                    self._remove_field_prefix(
                        field_name,
                        value,
                    )
                )

                fields[field_name] = value

        # -------------------------------------------------
        # Final cleanup
        # -------------------------------------------------

        for field_name in list(
            fields.keys()
        ):

            fields[field_name] = (
                self.clean_value(
                    fields[field_name]
                )
            )

        return fields