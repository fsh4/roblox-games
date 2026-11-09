"""Logging setup: readable console output plus a rotating file in the data folder."""
from __future__ import annotations

import logging
import logging.handlers
import re
from pathlib import Path

_TOKEN = re.compile(r"[MNO][A-Za-z\d_-]{23,25}\.[A-Za-z\d_-]{6}\.[A-Za-z\d_-]{27,40}")


class RedactingFilter(logging.Filter):
    """Never let something shaped like a Discord bot token reach a log line."""

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        if _TOKEN.search(msg):
            record.msg = _TOKEN.sub("[redacted-token]", msg)
            record.args = ()
        return True


def setup_logging(level: str, log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s", "%Y-%m-%d %H:%M:%S")
    root = logging.getLogger()
    root.setLevel(level)
    for h in list(root.handlers):
        root.removeHandler(h)
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    console.addFilter(RedactingFilter())
    root.addHandler(console)
    file_handler = logging.handlers.RotatingFileHandler(log_dir / "studio.log", maxBytes=5_000_000, backupCount=3, encoding="utf-8")
    file_handler.setFormatter(fmt)
    file_handler.addFilter(RedactingFilter())
    root.addHandler(file_handler)
    logging.getLogger("discord").setLevel(max(logging.INFO, root.level))
    return logging.getLogger("studio")
