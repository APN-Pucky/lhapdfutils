from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from lhagrid import lhapdf as gridpdf

from ._combine import (
    _build_auto_grid_defaults,
    _resolve_pdfsets,
    construct_envelop_grids,
)

DEFAULT_X_POINTS = 50
DEFAULT_Q_POINTS = 30
DEFAULT_X_SCALE = "linear"
DEFAULT_Q_SCALE = "log"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lhapdf-combine",
        description=(
            "Construct an LHAPDF envelope set from one or more existing LHAPDF sets."
        ),
        epilog=(
            "Examples:\n"
            "  lhapdf-combine BFG_I BFG_II --name BFG_ENV --set-index 999001\n"
            "  lhapdf-combine BFG_I BFG_II --x-min 1e-4 --x-max 1.0 --x-points 80 \\\n"
            "      --q-min 2.0 --q-max 200.0 --q-points 40 --q-scale log \\\n"
            "      --flavors 21,1,2,3,4,5 --install-dir ./pdfsets\n"
            "  lhapdf-combine BFG_I BFG_II --config combine.json --install-dir ./pdfsets"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "pdfsets",
        nargs="+",
        help="Input LHAPDF set names to combine.",
    )
    parser.add_argument(
        "--config",
        type=str,
        help=(
            "Path to a JSON file containing gridding information. Use '-' to read "
            "the JSON document from stdin."
        ),
    )
    parser.add_argument(
        "--name",
        type=str,
        help="Output LHAPDF set name. Defaults to a name derived from the inputs.",
    )
    parser.add_argument(
        "--set-index",
        type=int,
        help="Output LHAPDF SetIndex.",
    )
    parser.add_argument(
        "--description",
        type=str,
        help="SetDesc for the generated set.",
    )
    parser.add_argument(
        "--authors",
        type=str,
        help="Authors field for the generated set.",
    )
    parser.add_argument(
        "--reference",
        type=str,
        help="Reference field for the generated set.",
    )
    parser.add_argument(
        "--data-version",
        type=str,
        help="DataVersion for the generated set.",
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
    parser.add_argument(
        "--x-grid",
        type=_parse_float_list,
        help="Comma-separated x-axis values for a single generated subgrid.",
    )
    parser.add_argument(
        "--x-min",
        type=float,
        help="Minimum x value when generating the x axis.",
    )
    parser.add_argument(
        "--x-max",
        type=float,
        help="Maximum x value when generating the x axis.",
    )
    parser.add_argument(
        "--x-points",
        type=_positive_int,
        help=f"Number of generated x samples. Defaults to {DEFAULT_X_POINTS}.",
    )
    parser.add_argument(
        "--x-scale",
        choices=("linear", "log"),
        help=f"Spacing used to generate the x axis. Defaults to {DEFAULT_X_SCALE}.",
    )
    parser.add_argument(
        "--q-grid",
        type=_parse_float_list,
        help="Comma-separated Q-axis values for a single generated subgrid.",
    )
    parser.add_argument(
        "--q-min",
        type=float,
        help="Minimum Q value when generating the Q axis.",
    )
    parser.add_argument(
        "--q-max",
        type=float,
        help="Maximum Q value when generating the Q axis.",
    )
    parser.add_argument(
        "--q-points",
        type=_positive_int,
        help=f"Number of generated Q samples. Defaults to {DEFAULT_Q_POINTS}.",
    )
    parser.add_argument(
        "--q-scale",
        choices=("linear", "log"),
        help=f"Spacing used to generate the Q axis. Defaults to {DEFAULT_Q_SCALE}.",
    )
    parser.add_argument(
        "--flavors",
        type=_parse_int_list,
        help=(
            "Comma-separated list of requested flavors for a single generated "
            "subgrid. Defaults to the common flavors shared by the input sets."
        ),
    )
    return parser


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("Value must be a positive integer.")
    return parsed


def _parse_float_list(value: str) -> list[float]:
    entries = _split_comma_separated_values(value)
    try:
        return [float(entry) for entry in entries]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Expected a comma-separated list of floating-point values."
        ) from exc


def _parse_int_list(value: str) -> list[int]:
    entries = _split_comma_separated_values(value)
    try:
        return [int(entry) for entry in entries]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Expected a comma-separated list of integer values."
        ) from exc


def _split_comma_separated_values(value: str) -> list[str]:
    entries = [entry.strip() for entry in value.split(",")]
    filtered_entries = [entry for entry in entries if entry]
    if not filtered_entries:
        raise argparse.ArgumentTypeError("Expected at least one comma-separated value.")
    return filtered_entries


def _load_config(path: str) -> Any:
    if path == "-":
        return json.load(argparse.FileType("r")(path))
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


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


def _build_axis(
    explicit_values: Sequence[float] | None,
    lower: float,
    upper: float,
    points: int,
    scale: str,
    axis_name: str,
) -> list[float]:
    if explicit_values is not None:
        return [float(value) for value in explicit_values]

    if lower >= upper:
        raise ValueError(f"{axis_name} axis requires {axis_name}-min < {axis_name}-max.")
    if scale == "log" and lower <= 0.0:
        raise ValueError(f"{axis_name} axis requires positive bounds for log spacing.")

    if scale == "log":
        return np.geomspace(lower, upper, points).tolist()
    return np.linspace(lower, upper, points).tolist()


def _validate_axis_arguments(
    parser: argparse.ArgumentParser,
    grid: Sequence[float] | None,
    lower: float | None,
    upper: float | None,
    points: int | None,
    scale: str | None,
    axis_name: str,
) -> None:
    if grid is None:
        return
    if any(value is not None for value in (lower, upper, points, scale)):
        parser.error(
            f"Use either --{axis_name}-grid or the generated-axis options for {axis_name}, not both."
        )


def _metadata_overrides(args: argparse.Namespace) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    if args.name is not None:
        overrides["name"] = args.name
    if args.set_index is not None:
        overrides["set_index"] = args.set_index
    if args.description is not None:
        overrides["set_desc"] = args.description
    if args.authors is not None:
        overrides["authors"] = args.authors
    if args.reference is not None:
        overrides["reference"] = args.reference
    if args.data_version is not None:
        overrides["data_version"] = args.data_version
    return overrides


def _build_gridding_information(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> Any:
    metadata = _metadata_overrides(args)
    auto_grid_requested = all(
        value is None
        for value in (
            args.x_grid,
            args.x_min,
            args.x_max,
            args.x_points,
            args.x_scale,
            args.q_grid,
            args.q_min,
            args.q_max,
            args.q_points,
            args.q_scale,
            args.flavors,
        )
    )

    if args.config is not None:
        if any(
            value is not None
            for value in (
                args.x_grid,
                args.x_min,
                args.x_max,
                args.x_points,
                args.x_scale,
                args.q_grid,
                args.q_min,
                args.q_max,
                args.q_points,
                args.q_scale,
                args.flavors,
            )
        ):
            parser.error(
                "Use either --config or the single-subgrid axis/flavor options, not both."
            )

        gridding_information = _load_config(args.config)
        if metadata:
            if not isinstance(gridding_information, dict):
                parser.error(
                    "Metadata overrides such as --name require the JSON config to be an object."
                )
            gridding_information = {**gridding_information, **metadata}
        return gridding_information

    if auto_grid_requested:
        if metadata:
            return metadata
        return None

    _validate_axis_arguments(
        parser,
        grid=args.x_grid,
        lower=args.x_min,
        upper=args.x_max,
        points=args.x_points,
        scale=args.x_scale,
        axis_name="x",
    )
    _validate_axis_arguments(
        parser,
        grid=args.q_grid,
        lower=args.q_min,
        upper=args.q_max,
        points=args.q_points,
        scale=args.q_scale,
        axis_name="q",
    )

    defaults = _build_auto_grid_defaults(_resolve_pdfsets(args.pdfsets))
    x_axis = _build_axis(
        explicit_values=args.x_grid,
        lower=args.x_min if args.x_min is not None else defaults.x_min,
        upper=args.x_max if args.x_max is not None else defaults.x_max,
        points=args.x_points if args.x_points is not None else DEFAULT_X_POINTS,
        scale=args.x_scale if args.x_scale is not None else DEFAULT_X_SCALE,
        axis_name="x",
    )
    q_axis = _build_axis(
        explicit_values=args.q_grid,
        lower=args.q_min if args.q_min is not None else defaults.q_min,
        upper=args.q_max if args.q_max is not None else defaults.q_max,
        points=args.q_points if args.q_points is not None else DEFAULT_Q_POINTS,
        scale=args.q_scale if args.q_scale is not None else DEFAULT_Q_SCALE,
        axis_name="q",
    )

    gridding_information = {
        **metadata,
        "subgrids": [
            {
                "x_axis": x_axis,
                "q_axis": q_axis,
                "flavor_axis": (
                    args.flavors
                    if args.flavors is not None
                    else defaults.flavor_axis
                ),
            }
        ],
    }
    return gridding_information


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        gridding_information = _build_gridding_information(parser, args)
        install_dir = _resolve_install_dir(args.install_dir)
        result = construct_envelop_grids(args.pdfsets, gridding_information)
        # Resolve the destination explicitly so we don't rely on downstream fallback logic.
        gridpdf.install(result, folder=install_dir)
    except Exception as exc:
        parser.exit(1, f"error: {exc}\n")

    print(f"Installed {result.name} to {install_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
