"""Application entry point."""

from __future__ import annotations

import argparse
import logging
import sys
from types import TracebackType

from PyQt6.QtWidgets import QApplication

from neurograph.gui import MainWindow, install_theme


def _log_uncaught(
    exc_type: type[BaseException], exc: BaseException, tb: TracebackType | None
) -> None:
    """Log unhandled exceptions instead of letting PyQt abort the whole process."""
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc, tb)
        return
    logging.getLogger("neurograph").error("Unhandled exception", exc_info=(exc_type, exc, tb))


def main() -> int:
    """Launch the NeuroGraph desktop application."""
    parser = argparse.ArgumentParser(description="Explore and manage a NeuroGraph knowledge graph.")
    parser.add_argument("--database", help="SQLite database path (defaults to per-user app data).")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    sys.excepthook = _log_uncaught
    application = QApplication(sys.argv[:1])
    application.setOrganizationName("NeuroGraph")
    application.setApplicationName("NeuroGraph")
    install_theme(application)
    window = MainWindow(args.database)
    window.show()
    return application.exec()


if __name__ == "__main__":
    raise SystemExit(main())
