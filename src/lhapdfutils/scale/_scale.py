from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from lhagrid import LHAInfo, LHAGrid, LHASet

if TYPE_CHECKING:
    import lhapdf


@dataclass(frozen=True)
class InputSpec:
    set_name: str
    member: int | None = None


@dataclass(frozen=True)
class OutputSpec:
    name: str
    set_index: int | None = None


def parse_input_spec(value: str) -> InputSpec:
    set_name, _, member_text = value.rpartition("/")
    if not set_name:
        return InputSpec(set_name=value)

    try:
        member = int(member_text)
    except ValueError:
        return InputSpec(set_name=value)

    if member < 0:
        raise ValueError("Selected member must be a non-negative integer.")
    return InputSpec(set_name=set_name, member=member)


def parse_output_spec(value: str) -> OutputSpec:
    name, _, set_index_text = value.rpartition("@")
    if not name:
        return OutputSpec(name=value)

    try:
        set_index = int(set_index_text)
    except ValueError:
        return OutputSpec(name=value)

    if set_index < 0:
        raise ValueError("Output SetIndex must be a non-negative integer.")
    return OutputSpec(name=name, set_index=set_index)


def build_scaled_pdfset(
    input_spec: str | InputSpec,
    output_spec: str | OutputSpec,
    *,
    factor: float = 1.0,
    only: Sequence[int] | None = None,
    exclude: Sequence[int] | None = None,
) -> LHASet:
    resolved_input = (
        parse_input_spec(input_spec) if isinstance(input_spec, str) else input_spec
    )
    resolved_output = (
        parse_output_spec(output_spec) if isinstance(output_spec, str) else output_spec
    )

    if only is not None and exclude is not None:
        raise ValueError("Use either only or exclude, not both.")

    source_pdfset = _resolve_pdfset(resolved_input.set_name)
    source_members = int(source_pdfset.size)

    if resolved_input.member is None:
        member_indices = list(range(source_members))
    else:
        if resolved_input.member >= source_members:
            raise ValueError(
                f"Member {resolved_input.member} is out of range for {resolved_input.set_name}; "
                f"expected 0 <= member < {source_members}."
            )
        member_indices = [resolved_input.member]

    available_flavors = _extract_source_flavors(source_pdfset)
    selected_flavors = _select_flavors(available_flavors, only=only, exclude=exclude)
    source_folder = _resolve_pdfset_folder(resolved_input.set_name)

    scaled_grids = []
    for member_index in member_indices:
        source_grid = LHAGrid.from_file(
            str(source_folder / f"{resolved_input.set_name}_{member_index:04d}.dat"),
            validate=False,
        )
        scaled_grids.append(_scale_grid(source_grid, factor, selected_flavors))

    return LHASet(
        name=resolved_output.name,
        info=_build_scaled_info(
            source_pdfset=source_pdfset,
            source_input=resolved_input,
            output=resolved_output,
            num_members=len(scaled_grids),
            factor=factor,
            selected_flavors=selected_flavors,
        ),
        grids=scaled_grids,
    )


def _resolve_pdfset(set_name: str) -> "lhapdf.PDFSet":
    try:
        import lhapdf
    except ImportError as exc:
        raise ImportError("LHAPDF is required to scale PDF sets.") from exc

    return lhapdf.getPDFSet(set_name)


def _resolve_pdfset_folder(set_name: str) -> Path:
    try:
        import lhapdf
    except ImportError as exc:
        raise ImportError("LHAPDF is required to scale PDF sets.") from exc

    search_paths = [Path(path) for path in lhapdf.paths()]
    for base_path in search_paths:
        candidate = base_path / set_name
        if candidate.is_dir() and (candidate / f"{set_name}.info").is_file():
            return candidate

    raise ValueError(f"Could not locate installed LHAPDF set {set_name}.")


def _extract_source_flavors(pdfset: "lhapdf.PDFSet") -> list[int]:
    if pdfset.has_key("Flavors"):
        value = pdfset.get_entry("Flavors")
        if isinstance(value, str):
            import ast

            value = ast.literal_eval(value)
        flavors = [int(entry) for entry in value]
    else:
        members = pdfset.mkPDFs()
        if not members:
            raise ValueError(f"PDF set {pdfset.name} does not contain any members.")
        flavors = [int(entry) for entry in members[0].flavors()]

    if not flavors:
        raise ValueError(f"PDF set {pdfset.name} does not define any flavors.")
    return flavors


def _select_flavors(
    available_flavors: Sequence[int],
    *,
    only: Sequence[int] | None,
    exclude: Sequence[int] | None,
) -> list[int]:
    available_set = set(available_flavors)

    if only is not None:
        selected = {int(flavor) for flavor in only}
        unknown = sorted(selected - available_set)
        if unknown:
            raise ValueError(f"Requested flavors are not present in the source set: {unknown}")
        return [int(flavor) for flavor in available_flavors if int(flavor) in selected]

    if exclude is not None:
        excluded = {int(flavor) for flavor in exclude}
        unknown = sorted(excluded - available_set)
        if unknown:
            raise ValueError(f"Excluded flavors are not present in the source set: {unknown}")
        selected = [int(flavor) for flavor in available_flavors if int(flavor) not in excluded]
        if not selected:
            raise ValueError("Excluding these flavors would leave nothing to scale.")
        return selected

    return [int(flavor) for flavor in available_flavors]


def _scale_grid(grid: LHAGrid, factor: float, selected_flavors: Sequence[int]) -> LHAGrid:
    selected_set = {int(flavor) for flavor in selected_flavors}
    scaled_subgrids = []

    for subgrid in grid.subgrids:
        scaled_rows = []
        for row in subgrid.data:
            scaled_rows.append(
                [
                    float(value) * factor if int(flavor) in selected_set else float(value)
                    for flavor, value in zip(subgrid.flavor_axis, row)
                ]
            )

        scaled_subgrids.append(
            LHAGrid.SubGridBlock(
                x_axis=[float(value) for value in subgrid.x_axis],
                q_axis=[float(value) for value in subgrid.q_axis],
                flavor_axis=[int(value) for value in subgrid.flavor_axis],
                data=scaled_rows,
            )
        )

    return LHAGrid(
        PdfType=grid.PdfType,
        Format=grid.Format,
        subgrids=scaled_subgrids,
    )


def _build_scaled_info(
    *,
    source_pdfset: "lhapdf.PDFSet",
    source_input: InputSpec,
    output: OutputSpec,
    num_members: int,
    factor: float,
    selected_flavors: Sequence[int],
) -> LHAInfo:
    source_name = source_input.set_name
    source_flavors = _extract_source_flavors(source_pdfset)
    set_desc = _format_set_desc(
        source_pdfset=source_pdfset,
        source_name=source_name,
        selected_member=source_input.member,
        factor=factor,
        selected_flavors=selected_flavors,
    )
    reference = _format_reference(
        source_pdfset=source_pdfset,
        source_name=source_name,
        factor=factor,
        selected_flavors=selected_flavors,
    )

    return LHAInfo(
        SetDesc=set_desc,
        SetIndex=output.set_index
        if output.set_index is not None
        else _get_int_entry(source_pdfset, "SetIndex", 0),
        Authors=_get_string_entry(source_pdfset, "Authors", "lhapdf-scale"),
        Reference=reference,
        Format=_get_string_entry(source_pdfset, "Format", "lhagrid1"),
        DataVersion=_get_string_entry(source_pdfset, "DataVersion", "1"),
        NumMembers=num_members,
        Particle=_get_int_entry(source_pdfset, "Particle", 2212),
        Flavors=source_flavors,
        OrderQCD=_get_int_entry(source_pdfset, "OrderQCD", 0),
        FlavorScheme=_get_string_entry(source_pdfset, "FlavorScheme", "unknown"),
        NumFlavors=_get_int_entry(
            source_pdfset,
            "NumFlavors",
            len([flavor for flavor in source_flavors if abs(flavor) <= 6 and flavor != 21]),
        ),
        ErrorType=_get_optional_string_entry(source_pdfset, "ErrorType")
        if num_members > 1
        else None,
        ForcePositive=_get_optional_int_entry(source_pdfset, "ForcePositive"),
        XMin=_get_float_entry(source_pdfset, "XMin", 0.0),
        XMax=_get_float_entry(source_pdfset, "XMax", 1.0),
        QMin=_get_float_entry(source_pdfset, "QMin", 1.0),
        QMax=_get_float_entry(source_pdfset, "QMax", 1.0),
        MZ=_get_float_entry(source_pdfset, "MZ", 91.1876),
        MUp=_get_float_entry(source_pdfset, "MUp", 0.0),
        MDown=_get_float_entry(source_pdfset, "MDown", 0.0),
        MStrange=_get_float_entry(source_pdfset, "MStrange", 0.0),
        MCharm=_get_float_entry(source_pdfset, "MCharm", 0.0),
        MBottom=_get_float_entry(source_pdfset, "MBottom", 0.0),
        MTop=_get_float_entry(source_pdfset, "MTop", 0.0),
        AlphaS_MZ=_get_optional_float_entry(source_pdfset, "AlphaS_MZ"),
        AlphaS_OrderQCD=_get_optional_int_entry(source_pdfset, "AlphaS_OrderQCD"),
        AlphaS_Type=_get_string_entry(source_pdfset, "AlphaS_Type", "unknown"),
        AlphaS_Qs=_get_optional_list_entry(source_pdfset, "AlphaS_Qs"),
        AlphaS_Vals=_get_optional_list_entry(source_pdfset, "AlphaS_Vals"),
        AlphaS_Lambda3=_get_optional_float_entry(source_pdfset, "AlphaS_Lambda3"),
        AlphaS_Lambda4=_get_optional_float_entry(source_pdfset, "AlphaS_Lambda4"),
        AlphaS_Lambda5=_get_optional_float_entry(source_pdfset, "AlphaS_Lambda5"),
        Interpolator=_get_optional_string_entry(source_pdfset, "Interpolator"),
    )


def _format_set_desc(
    *,
    source_pdfset: "lhapdf.PDFSet",
    source_name: str,
    selected_member: int | None,
    factor: float,
    selected_flavors: Sequence[int],
) -> str:
    source_desc = _get_string_entry(source_pdfset, "SetDesc", source_name)
    member_suffix = (
        f", member {selected_member}" if selected_member is not None else ", all members"
    )
    flavor_suffix = ", ".join(str(flavor) for flavor in selected_flavors)
    return (
        f"{source_desc} [scaled from {source_name}{member_suffix} by factor {factor:g} "
        f"for flavors {flavor_suffix}]"
    )


def _format_reference(
    *,
    source_pdfset: "lhapdf.PDFSet",
    source_name: str,
    factor: float,
    selected_flavors: Sequence[int],
) -> str:
    source_reference = _get_optional_string_entry(source_pdfset, "Reference")
    generated_note = (
        f"Generated with lhapdf-scale from {source_name}; factor={factor:g}; "
        f"flavors={','.join(str(flavor) for flavor in selected_flavors)}"
    )
    if source_reference and source_reference != "None":
        return f"{source_reference}; {generated_note}"
    return generated_note


def _get_string_entry(pdfset: "lhapdf.PDFSet", key: str, default: str) -> str:
    if pdfset.has_key(key):
        return str(pdfset.get_entry(key))
    return default


def _get_optional_string_entry(
    pdfset: "lhapdf.PDFSet", key: str
) -> str | None:
    if pdfset.has_key(key):
        return str(pdfset.get_entry(key))
    return None


def _get_int_entry(pdfset: "lhapdf.PDFSet", key: str, default: int) -> int:
    if pdfset.has_key(key):
        return int(pdfset.get_entry(key))
    return default


def _get_float_entry(pdfset: "lhapdf.PDFSet", key: str, default: float) -> float:
    if pdfset.has_key(key):
        return float(pdfset.get_entry(key))
    return default


def _get_optional_int_entry(pdfset: "lhapdf.PDFSet", key: str) -> int | None:
    if pdfset.has_key(key):
        return int(pdfset.get_entry(key))
    return None


def _get_optional_float_entry(pdfset: "lhapdf.PDFSet", key: str) -> float | None:
    if pdfset.has_key(key):
        return float(pdfset.get_entry(key))
    return None


def _get_optional_list_entry(
    pdfset: "lhapdf.PDFSet", key: str
) -> list[float] | None:
    if not pdfset.has_key(key):
        return None

    value = pdfset.get_entry(key)
    if isinstance(value, str):
        import ast

        value = ast.literal_eval(value)

    return [float(entry) for entry in value]
