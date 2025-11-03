"""Utilities for accessing physionet data via wfdb
"""

from pathlib import Path
import wfdb


def init_wfdb(data_root):
    """Must be called to enable local data access
    """
    data_root = Path(data_root)
    wfdb.set_db_index_url(data_root.as_uri())
    wfdb.reset_data_sources()
    local_data_source = wfdb.DataSource(
        'local', wfdb.DataSourceType.LOCAL, data_root.as_posix())
    wfdb.add_data_source(local_data_source)
