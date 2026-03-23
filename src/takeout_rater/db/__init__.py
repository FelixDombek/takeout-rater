"""Database package: SQLite schema, migrations, and query helpers.

Public API
----------
- :func:`~takeout_rater.db.connection.open_library_db` — open / create the DB
- :func:`~takeout_rater.db.connection.library_state_dir` — locate the state dir
- :func:`~takeout_rater.db.queries.upsert_asset` — insert or update an asset
- :func:`~takeout_rater.db.queries.get_asset_by_id` — fetch one asset by ID
- :func:`~takeout_rater.db.queries.list_assets` — paginated asset listing
- :func:`~takeout_rater.db.queries.count_assets` — total matching asset count
- :class:`~takeout_rater.db.queries.AssetRow` — typed asset row dataclass
- :data:`~takeout_rater.db.queries.CURRENT_INDEXER_VERSION` — pipeline version constant
- :func:`~takeout_rater.db.queries.list_asset_ids_needing_rescan` — rescan candidates
- :func:`~takeout_rater.db.queries.count_assets_needing_rescan` — stale asset count

See ``docs/agents/db-guidelines.md`` for conventions.
"""

from takeout_rater.db.connection import library_state_dir, open_library_db
from takeout_rater.db.queries import (
    CURRENT_INDEXER_VERSION,
    AssetRow,
    count_assets,
    count_assets_needing_rescan,
    get_asset_by_id,
    list_asset_ids_needing_rescan,
    list_assets,
    upsert_asset,
)

__all__ = [
    "CURRENT_INDEXER_VERSION",
    "AssetRow",
    "count_assets",
    "count_assets_needing_rescan",
    "get_asset_by_id",
    "library_state_dir",
    "list_asset_ids_needing_rescan",
    "list_assets",
    "open_library_db",
    "upsert_asset",
]
