"""
MRI preprocessing pipeline for in-vivo and ex-vivo image pairs.
Provides tools for image alignment, normalization, and super-resolution data preparation.
"""

__version__ = "1.0.0"


# Set up package-level logger
import logging

from preprocessing.models import ImagePair, ProcessingResult

logging.getLogger(__name__).addHandler(logging.NullHandler())
