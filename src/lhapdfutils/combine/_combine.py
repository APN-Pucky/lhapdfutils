from __future__ import annotations

import ast
import math
from functools import lru_cache
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from lhagrid import LHAInfo, LHAGrid, LHASet

if TYPE_CHECKING:
    import lhapdf


@dataclass
class SubgridDefinition:
    x_axis: Sequence[float]
    q_axis: Sequence[float]
    flavor_axis: Sequence[int]


@dataclass
class GriddingInformation:
    subgrids: Sequence[SubgridDefinition]
    name: Optional[str] = None
    set_desc: Optional[str] = None
    set_index: int = 0
    authors: Optional[str] = None
    reference: Optional[str] = None
    data_version: Optional[str] = None


@dataclass
class _PointSummary:
    central: float
    upper: float
    lower: float


def construct_envelop_grids(
    pdfsets: Sequence[Any], gridding_information: Any = None
) -> LHASet:
    """
    Construct an envelope PDF grid set.

    The returned ``lhagrid.LHASet`` contains three members:

    - member 0: average of the per-set central values
    - member 1: upper envelope across the per-set uncertainty bands
    - member 2: lower envelope across the per-set uncertainty bands
    """

    resolved_pdfsets = _resolve_pdfsets(pdfsets)
    pdf_members = [pdfset.mkPDFs() for pdfset in resolved_pdfsets]
    normalized_gridding = _resolve_gridding_information(
        resolved_pdfsets, pdf_members, gridding_information
    )
    output_name = normalized_gridding.name or _default_output_name(resolved_pdfsets)
    _validate_requested_flavors(resolved_pdfsets, pdf_members, normalized_gridding)

    central_subgrids = []
    upper_subgrids = []
    lower_subgrids = []

    for subgrid_definition in normalized_gridding.subgrids:
        central_data = []
        upper_data = []
        lower_data = []

        for x_value in subgrid_definition.x_axis:
            for q_value in subgrid_definition.q_axis:
                central_row = []
                upper_row = []
                lower_row = []

                for flavor in subgrid_definition.flavor_axis:
                    summaries = [
                        _summarize_point(pdfset, members, flavor, x_value, q_value)
                        for pdfset, members in zip(resolved_pdfsets, pdf_members)
                    ]
                    central_row.append(
                        sum(summary.central for summary in summaries) / len(summaries)
                    )
                    upper_row.append(max(summary.upper for summary in summaries))
                    lower_row.append(min(summary.lower for summary in summaries))

                central_data.append(central_row)
                upper_data.append(upper_row)
                lower_data.append(lower_row)

        x_axis = [float(value) for value in subgrid_definition.x_axis]
        q_axis = [float(value) for value in subgrid_definition.q_axis]
        flavor_axis = [int(value) for value in subgrid_definition.flavor_axis]

        central_subgrids.append(
            LHAGrid.SubGridBlock(x_axis, q_axis, flavor_axis, central_data)
        )
        upper_subgrids.append(
            LHAGrid.SubGridBlock(x_axis, q_axis, flavor_axis, upper_data)
        )
        lower_subgrids.append(
            LHAGrid.SubGridBlock(x_axis, q_axis, flavor_axis, lower_data)
        )

    info = _build_info(resolved_pdfsets, normalized_gridding)

    return LHASet(
        name=output_name,
        info=info,
        grids=[
            LHAGrid("central", info.Format, central_subgrids),
            LHAGrid("error", info.Format, upper_subgrids),
            LHAGrid("error", info.Format, lower_subgrids),
        ],
    )


def construct_envelope_grids(
    pdfsets: Sequence[Any], gridding_information: Any = None
) -> LHASet:
    """
    Alias for :func:`construct_envelop_grids`.
    """

    return construct_envelop_grids(pdfsets, gridding_information)


@dataclass(frozen=True)
class _AutoGridDefaults:
    x_min: float
    x_max: float
    q_min: float
    q_max: float
    x_axis: list[float]
    q_axis: list[float]
    flavor_axis: list[int]


def _resolve_pdfsets(pdfsets: Sequence[Any]) -> list["lhapdf.PDFSet"]:
    if not pdfsets:
        raise ValueError("At least one PDF set is required.")

    try:
        import lhapdf
    except ImportError as exc:
        raise ImportError(
            "LHAPDF is required to construct envelope grids."
        ) from exc

    resolved_pdfsets = []
    for pdfset in pdfsets:
        if isinstance(pdfset, str):
            resolved_pdfsets.append(lhapdf.getPDFSet(pdfset))
            continue

        if isinstance(pdfset, lhapdf.PDFSet):
            resolved_pdfsets.append(pdfset)
            continue

        raise TypeError(
            "Each item in pdfsets must be an LHAPDF set name or an lhapdf.PDFSet instance."
        )

    return resolved_pdfsets


def _resolve_gridding_information(
    pdfsets: Sequence["lhapdf.PDFSet"],
    pdf_members: Sequence[Sequence["lhapdf.PDF"]],
    gridding_information: Any,
) -> GriddingInformation:
    if gridding_information is None:
        return _build_auto_gridding_information(pdfsets, pdf_members)

    if (
        isinstance(gridding_information, Mapping)
        and _is_metadata_only_gridding_mapping(gridding_information)
    ):
        return _build_auto_gridding_information(
            pdfsets, pdf_members, metadata=gridding_information
        )

    return _normalize_gridding_information(gridding_information)


def _build_auto_gridding_information(
    pdfsets: Sequence["lhapdf.PDFSet"],
    pdf_members: Sequence[Sequence["lhapdf.PDF"]],
    metadata: Mapping[str, Any] | None = None,
) -> GriddingInformation:
    defaults = _build_auto_grid_defaults(pdfsets, pdf_members)
    metadata = metadata or {}

    return GriddingInformation(
        subgrids=[
            SubgridDefinition(
                x_axis=defaults.x_axis,
                q_axis=defaults.q_axis,
                flavor_axis=defaults.flavor_axis,
            )
        ],
        name=_optional_string(metadata, "name", "set_name"),
        set_desc=_optional_string(
            metadata, "set_desc", "description", "set_description"
        ),
        set_index=int(metadata.get("set_index", 0)),
        authors=_optional_string(metadata, "authors"),
        reference=_optional_string(metadata, "reference"),
        data_version=_optional_string(metadata, "data_version", "dataversion"),
    )


def _build_auto_grid_defaults(
    pdfsets: Sequence["lhapdf.PDFSet"],
    pdf_members: Sequence[Sequence["lhapdf.PDF"]] | None = None,
) -> _AutoGridDefaults:
    if not pdfsets:
        raise ValueError("At least one PDF set is required.")

    if pdf_members is None:
        pdf_members = [pdfset.mkPDFs() for pdfset in pdfsets]

    central_members = []
    for pdfset, members in zip(pdfsets, pdf_members):
        if not members:
            raise ValueError(
                f"PDF set {pdfset.name} does not contain any members to summarize."
            )
        central_members.append(members[0])

    x_min = max(float(member.xMin) for member in central_members)
    x_max = min(float(member.xMax) for member in central_members)
    q_min = max(math.sqrt(float(member.q2Min)) for member in central_members)
    q_max = min(math.sqrt(float(member.q2Max)) for member in central_members)

    if x_min >= x_max:
        raise ValueError(
            f"Input PDF sets do not have an overlapping x range: {x_min} >= {x_max}."
        )
    if q_min >= q_max:
        raise ValueError(
            f"Input PDF sets do not have an overlapping Q range: {q_min} >= {q_max}."
        )

    return _AutoGridDefaults(
        x_min=x_min,
        x_max=x_max,
        q_min=q_min,
        q_max=q_max,
        x_axis=_collect_common_axis_points(pdfsets, "x", x_min, x_max),
        q_axis=_collect_common_axis_points(pdfsets, "q", q_min, q_max),
        flavor_axis=_collect_common_flavors(central_members),
    )


def _collect_common_axis_points(
    pdfsets: Sequence["lhapdf.PDFSet"],
    axis_name: str,
    lower: float,
    upper: float,
) -> list[float]:
    candidate_values: list[float] = []
    for pdfset in pdfsets:
        reference_grid = _load_reference_grid(pdfset.name)
        for subgrid in reference_grid.subgrids:
            axis_values = subgrid.x_axis if axis_name == "x" else subgrid.q_axis
            candidate_values.extend(
                float(value)
                for value in axis_values
                if _value_within_bounds(float(value), lower, upper)
            )

    axis_values = _unique_sorted_values(candidate_values)
    if not axis_values:
        raise ValueError(
            f"Could not determine any native {axis_name} grid points in the common range."
        )

    _insert_boundary_if_missing(axis_values, lower)
    _insert_boundary_if_missing(axis_values, upper)
    return axis_values


def _collect_common_flavors(central_members: Sequence["lhapdf.PDF"]) -> list[int]:
    first_flavors = [int(flavor) for flavor in central_members[0].flavors()]
    common_flavors = set(first_flavors)
    for member in central_members[1:]:
        common_flavors &= {int(flavor) for flavor in member.flavors()}

    ordered_common_flavors = [
        flavor for flavor in first_flavors if flavor in common_flavors
    ]
    if not ordered_common_flavors:
        raise ValueError("Input PDF sets do not share any common flavors.")
    return ordered_common_flavors


@lru_cache(maxsize=None)
def _load_reference_grid(pdfset_name: str) -> LHAGrid:
    return LHAGrid.from_file(_resolve_reference_grid_path(pdfset_name), validate=False)


def _resolve_reference_grid_path(pdfset_name: str) -> str:
    try:
        import lhapdf
    except ImportError as exc:
        raise ImportError(
            "LHAPDF is required to inspect input sets and build the output grid."
        ) from exc

    candidate_paths = [Path(path) for path in lhapdf.paths()]
    for base_path in candidate_paths:
        candidate = base_path / pdfset_name / f"{pdfset_name}_0000.dat"
        if candidate.is_file():
            return str(candidate)

    raise ValueError(
        f"Could not locate the LHAPDF grid file for {pdfset_name} in {candidate_paths}."
    )


def _unique_sorted_values(values: Sequence[float]) -> list[float]:
    return sorted({float(value) for value in values})


def _insert_boundary_if_missing(values: list[float], boundary: float) -> None:
    if not any(math.isclose(value, boundary, rel_tol=1.0e-12, abs_tol=1.0e-15) for value in values):
        values.append(float(boundary))
        values.sort()


def _value_within_bounds(value: float, lower: float, upper: float) -> bool:
    return value >= lower - 1.0e-12 and value <= upper + 1.0e-12


def _is_metadata_only_gridding_mapping(gridding_information: Mapping[str, Any]) -> bool:
    structural_keys = {
        "subgrids",
        "x_axis",
        "x_grid",
        "q_axis",
        "q_grid",
        "flavor_axis",
        "flavors",
        "flavours",
    }
    if any(key in gridding_information for key in structural_keys):
        return False

    metadata_keys = {
        "name",
        "set_name",
        "set_desc",
        "description",
        "set_description",
        "set_index",
        "authors",
        "reference",
        "data_version",
        "dataversion",
    }
    return not gridding_information or any(
        key in gridding_information for key in metadata_keys
    )


def _normalize_gridding_information(
    gridding_information: Any,
) -> GriddingInformation:
    if isinstance(gridding_information, GriddingInformation):
        return GriddingInformation(
            subgrids=[
                _normalize_subgrid_definition(subgrid)
                for subgrid in gridding_information.subgrids
            ],
            name=gridding_information.name,
            set_desc=gridding_information.set_desc,
            set_index=gridding_information.set_index,
            authors=gridding_information.authors,
            reference=gridding_information.reference,
            data_version=gridding_information.data_version,
        )

    if isinstance(gridding_information, Mapping):
        subgrids_value = gridding_information.get("subgrids")
        if subgrids_value is None:
            subgrids = [_normalize_subgrid_definition(gridding_information)]
        else:
            subgrids = [
                _normalize_subgrid_definition(subgrid) for subgrid in subgrids_value
            ]

        return GriddingInformation(
            subgrids=subgrids,
            name=_optional_string(gridding_information, "name", "set_name"),
            set_desc=_optional_string(
                gridding_information, "set_desc", "description", "set_description"
            ),
            set_index=int(gridding_information.get("set_index", 0)),
            authors=_optional_string(gridding_information, "authors"),
            reference=_optional_string(gridding_information, "reference"),
            data_version=_optional_string(
                gridding_information, "data_version", "dataversion"
            ),
        )

    if _is_non_string_iterable(gridding_information):
        return GriddingInformation(
            subgrids=[
                _normalize_subgrid_definition(subgrid)
                for subgrid in gridding_information
            ]
        )

    raise TypeError(
        "gridding_information must be a GriddingInformation instance, a mapping, or an iterable of subgrid definitions."
    )


def _normalize_subgrid_definition(subgrid_definition: Any) -> SubgridDefinition:
    if isinstance(subgrid_definition, SubgridDefinition):
        x_axis = [float(value) for value in subgrid_definition.x_axis]
        q_axis = [float(value) for value in subgrid_definition.q_axis]
        flavor_axis = [int(value) for value in subgrid_definition.flavor_axis]
        _validate_subgrid_axes(x_axis, q_axis, flavor_axis)
        return SubgridDefinition(x_axis=x_axis, q_axis=q_axis, flavor_axis=flavor_axis)

    if not isinstance(subgrid_definition, Mapping):
        raise TypeError(
            "Each subgrid definition must be a SubgridDefinition or a mapping."
        )

    x_axis = [float(value) for value in _required_value(subgrid_definition, "x_axis", "x_grid")]
    q_axis = [float(value) for value in _required_value(subgrid_definition, "q_axis", "q_grid")]
    flavor_axis = [
        int(value)
        for value in _required_value(
            subgrid_definition, "flavor_axis", "flavors", "flavours"
        )
    ]

    _validate_subgrid_axes(x_axis, q_axis, flavor_axis)
    return SubgridDefinition(x_axis=x_axis, q_axis=q_axis, flavor_axis=flavor_axis)


def _validate_subgrid_axes(
    x_axis: Sequence[float], q_axis: Sequence[float], flavor_axis: Sequence[int]
) -> None:
    if not x_axis:
        raise ValueError("Each subgrid must define a non-empty x_axis.")
    if not q_axis:
        raise ValueError("Each subgrid must define a non-empty q_axis.")
    if not flavor_axis:
        raise ValueError("Each subgrid must define a non-empty flavor_axis.")
    if any(x_value < 0.0 or x_value > 1.0 for x_value in x_axis):
        raise ValueError("All x_axis entries must satisfy 0 <= x <= 1.")
    if any(q_value < 0.0 for q_value in q_axis):
        raise ValueError("All q_axis entries must be positive.")


def _validate_requested_flavors(
    pdfsets: Sequence["lhapdf.PDFSet"],
    pdf_members: Sequence[Sequence["lhapdf.PDF"]],
    gridding_information: GriddingInformation,
) -> None:
    requested_flavors = {
        flavor
        for subgrid in gridding_information.subgrids
        for flavor in subgrid.flavor_axis
    }

    for pdfset, members in zip(pdfsets, pdf_members):
        missing_flavors = sorted(
            flavor for flavor in requested_flavors if not members[0].hasFlavor(flavor)
        )
        if missing_flavors:
            raise ValueError(
                f"PDF set {pdfset.name} does not provide the requested flavors: {missing_flavors}"
            )


def _summarize_point(
    pdfset: "lhapdf.PDFSet",
    members: Sequence["lhapdf.PDF"],
    flavor: int,
    x_value: float,
    q_value: float,
) -> _PointSummary:
    member_values = [member.xfxQ(flavor, x_value, q_value) for member in members]
    if len(member_values) == 0:
        raise ValueError(
            f"PDF set {pdfset.name} does not contain any members to summarize."
        )
    if len(member_values) == 1:
        return _PointSummary(central=float(member_values[0]), upper=float(member_values[0]), lower=float(member_values[0]))
    uncertainty = pdfset.uncertainty(member_values)

    return _PointSummary(
        central=float(uncertainty.central),
        upper=float(uncertainty.central + uncertainty.errplus),
        lower=float(uncertainty.central - uncertainty.errminus),
    )


def _build_info(
    pdfsets: Sequence["lhapdf.PDFSet"],
    gridding_information: GriddingInformation,
) -> LHAInfo:
    first_pdfset = pdfsets[0]
    input_names = [pdfset.name for pdfset in pdfsets]
    all_flavors = sorted(
        {
            flavor
            for subgrid in gridding_information.subgrids
            for flavor in subgrid.flavor_axis
        }
    )
    all_x_values = [
        x_value for subgrid in gridding_information.subgrids for x_value in subgrid.x_axis
    ]
    all_q_values = [
        q_value for subgrid in gridding_information.subgrids for q_value in subgrid.q_axis
    ]
    x_min, x_max = _range_from_grid_or_pdfset(first_pdfset, all_x_values, "XMin", "XMax")
    q_min, q_max = _range_from_grid_or_pdfset(first_pdfset, all_q_values, "QMin", "QMax")

    set_desc = gridding_information.set_desc or (
        f"Envelope combination of {', '.join(input_names)}"
    )
    reference = gridding_information.reference or (
        f"Constructed with lhapdfcombine from {', '.join(input_names)}"
    )

    return LHAInfo(
        SetDesc=set_desc,
        SetIndex=gridding_information.set_index,
        Authors=gridding_information.authors or "lhapdfcombine",
        Reference=reference,
        Format=_get_string_entry(first_pdfset, "Format", "lhagrid1"),
        DataVersion=gridding_information.data_version
        or _get_string_entry(first_pdfset, "DataVersion", "1"),
        NumMembers=3,
        Particle=_get_int_entry(first_pdfset, "Particle", 2212),
        Flavors=all_flavors,
        OrderQCD=_get_int_entry(first_pdfset, "OrderQCD", 0),
        FlavorScheme=_get_string_entry(first_pdfset, "FlavorScheme", "unknown"),
        NumFlavors=_get_int_entry(
            first_pdfset,
            "NumFlavors",
            len([flavor for flavor in all_flavors if abs(flavor) <= 6 and flavor != 21]),
        ),
        ErrorType="hessian",
        Interpolator=_get_string_entry(first_pdfset, "Interpolator", "logcubic"),
        XMin=x_min,
        XMax=x_max,
        QMin=q_min,
        QMax=q_max,
        MZ=_get_float_entry(first_pdfset, "MZ", 91.1876),
        MUp=_get_float_entry(first_pdfset, "MUp", 0.0),
        MDown=_get_float_entry(first_pdfset, "MDown", 0.0),
        MStrange=_get_float_entry(first_pdfset, "MStrange", 0.0),
        MCharm=_get_float_entry(first_pdfset, "MCharm", 0.0),
        MBottom=_get_float_entry(first_pdfset, "MBottom", 0.0),
        MTop=_get_float_entry(first_pdfset, "MTop", 0.0),
        ForcePositive=_get_optional_int_entry(first_pdfset, "ForcePositive"),
        AlphaS_MZ=_get_optional_float_entry(first_pdfset, "AlphaS_MZ"),
        AlphaS_OrderQCD=_get_optional_int_entry(first_pdfset, "AlphaS_OrderQCD"),
        AlphaS_Type=_get_string_entry(first_pdfset, "AlphaS_Type", "unknown"),
        AlphaS_Qs=_get_optional_list_entry(first_pdfset, "AlphaS_Qs"),
        AlphaS_Vals=_get_optional_list_entry(first_pdfset, "AlphaS_Vals"),
        AlphaS_Lambda3=_get_optional_float_entry(first_pdfset, "AlphaS_Lambda3"),
        AlphaS_Lambda4=_get_optional_float_entry(first_pdfset, "AlphaS_Lambda4"),
        AlphaS_Lambda5=_get_optional_float_entry(first_pdfset, "AlphaS_Lambda5"),
    )


def _default_output_name(pdfsets: Sequence["lhapdf.PDFSet"]) -> str:
    joined_names = "_".join(pdfset.name for pdfset in pdfsets)
    if not joined_names:
        return "combined_envelope"
    candidate = f"{joined_names}_envelope"
    if len(candidate) <= 80:
        return candidate
    return "combined_envelope"


def _range_from_grid_or_pdfset(
    pdfset: "lhapdf.PDFSet",
    values: Sequence[float],
    minimum_key: str,
    maximum_key: str,
) -> tuple[float, float]:
    lower = min(values)
    upper = max(values)
    if lower < upper:
        return lower, upper

    fallback_lower = _get_float_entry(pdfset, minimum_key, lower)
    fallback_upper = _get_float_entry(pdfset, maximum_key, upper)
    if fallback_lower < fallback_upper:
        return fallback_lower, fallback_upper

    return lower, upper + max(abs(upper) * 1.0e-12, 1.0e-12)


def _required_value(mapping: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    joined_keys = ", ".join(keys)
    raise KeyError(f"Missing required key. Expected one of: {joined_keys}")


def _optional_string(mapping: Mapping[str, Any], *keys: str) -> Optional[str]:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return str(mapping[key])
    return None


def _get_string_entry(pdfset: "lhapdf.PDFSet", key: str, default: str) -> str:
    if pdfset.has_key(key):
        return str(pdfset.get_entry(key))
    return default


def _get_int_entry(pdfset: "lhapdf.PDFSet", key: str, default: int) -> int:
    if pdfset.has_key(key):
        return int(pdfset.get_entry(key))
    return default


def _get_float_entry(pdfset: "lhapdf.PDFSet", key: str, default: float) -> float:
    if pdfset.has_key(key):
        return float(pdfset.get_entry(key))
    return default


def _get_optional_int_entry(pdfset: "lhapdf.PDFSet", key: str) -> Optional[int]:
    if pdfset.has_key(key):
        return int(pdfset.get_entry(key))
    return None


def _get_optional_float_entry(pdfset: "lhapdf.PDFSet", key: str) -> Optional[float]:
    if pdfset.has_key(key):
        return float(pdfset.get_entry(key))
    return None


def _get_optional_list_entry(
    pdfset: "lhapdf.PDFSet", key: str
) -> Optional[list[float]]:
    if not pdfset.has_key(key):
        return None

    value = pdfset.get_entry(key)
    if isinstance(value, str):
        parsed_value = ast.literal_eval(value)
    else:
        parsed_value = value

    return [float(entry) for entry in parsed_value]


def _is_non_string_iterable(value: Any) -> bool:
    return isinstance(value, Iterable) and not isinstance(value, (str, bytes, Mapping))
