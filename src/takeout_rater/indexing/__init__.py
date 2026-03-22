"""Indexing package: Takeout scanner, sidecar parser, and thumbnail generator.

Public API
----------
- :func:`~takeout_rater.indexing.scanner.scan_takeout` — walk a Takeout tree
- :func:`~takeout_rater.indexing.sidecar.parse_sidecar` — parse a sidecar JSON
- :func:`~takeout_rater.indexing.thumbnailer.generate_thumbnail` — create a JPEG thumb
- :func:`~takeout_rater.indexing.thumbnailer.thumb_path_for_id` — compute thumb path
"""

from takeout_rater.indexing.scanner import AssetFile, scan_takeout
from takeout_rater.indexing.sidecar import SidecarData, parse_sidecar
from takeout_rater.indexing.thumbnailer import generate_thumbnail, thumb_path_for_id

__all__ = [
    "AssetFile",
    "SidecarData",
    "generate_thumbnail",
    "parse_sidecar",
    "scan_takeout",
    "thumb_path_for_id",
]
