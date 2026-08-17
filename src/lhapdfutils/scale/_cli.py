from __future__ import annotations

import argparse
import os
from collections.abc import Sequence
from pathlib import Path

from lhagrid import lhapdf as gridpdf

from ._scale import build_scaled_pdfset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lhapdf-scale",
        description="Scale selected flavors in an installed LHAPDF set.",
        epilog=(
            "Examples:\n"
            "  lhapdf-scale BFG_I BFG_I_x2@999900 --factor 2\n"
            "  lhapdf-scale BFG_I/0 BFG_I_gluon@999901 --factor 1.5 --only 21\n"
            "  lhapdf-scale BFG_GRV_ENV BFG_GRV_ENV_quarks@999902 --factor 0.5 --except 21"
            "  lhapdf-scale BFG BFG_only_photons@999902 --factor 1e-9 --except 22"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "input_spec",
        help="Source set as SET or SET/MEMBER.",
    )
    parser.add_argument(
        "output_spec",
        help="Output set as NAME or NAME@SETINDEX.",
    )
    parser.add_argument(
        "--factor",
        type=float,
        default=1.0,
        help="Multiplicative scale factor to apply. Defaults to 1.",
    )
    flavor_group = parser.add_mutually_exclusive_group()
    flavor_group.add_argument(
        "--only",
        type=_parse_int_list,
        help="Comma-separated PDG IDs to scale.",
    )
    flavor_group.add_argument(
        "--except",
        dest="exclude",
        type=_parse_int_list,
        help="Comma-separated PDG IDs to leave unchanged while scaling the rest.",
    )
    parser.add_argument(
        "--install-dir",
        type=str,
        help=(
            "Directory where the generated set should be installed. Defaults to the "
            "first writable LHAPDF search path when available, otherwise the current "
            "directory."
        ),
    )
    return parser


def _parse_int_list(value: str) -> list[int]:
    entries = [entry.strip() for entry in value.split(",")]
    filtered = [entry for entry in entries if entry]
    if not filtered:
        raise argparse.ArgumentTypeError("Expected at least one comma-separated integer.")

    try:
        return [int(entry) for entry in filtered]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Expected a comma-separated list of integers."
        ) from exc


def _resolve_install_dir(requested_install_dir: str | None) -> str:
    if requested_install_dir:
        return requested_install_dir

    try:
        import lhapdf
    except ImportError:
        return "."

    for path in map(str, lhapdf.paths()):
        if os.path.isdir(path) and os.access(path, os.W_OK):
            return path
        parent = Path(path).parent
        if not Path(path).exists() and parent.is_dir() and os.access(parent, os.W_OK):
            return path
    return "."


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        result = build_scaled_pdfset(
            args.input_spec,
            args.output_spec,
            factor=args.factor,
            only=args.only,
            exclude=args.exclude,
        )
        install_dir = _resolve_install_dir(args.install_dir)
        gridpdf.install(result, folder=install_dir)
    except Exception as exc:
        parser.exit(1, f"error: {exc}\n")

    print(f"Installed {result.name} to {install_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
