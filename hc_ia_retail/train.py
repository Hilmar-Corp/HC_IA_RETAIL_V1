from __future__ import annotations

# Re-export to match tests: import hc_ia_retail.train -> write_data_report / generate_data_report
from .audit import generate_data_report, write_data_report  # noqa: F401