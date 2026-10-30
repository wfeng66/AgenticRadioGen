#!/usr/bin/env python3
"""Browse a CT volume, or the CT series in a patient's DICOM folder.

Twenty consecutive slices per page: 5 columns x 4 rows.
Use the ORIGINAL CT with HU intensities, not a mask or normalized model input.
DICOM discovery supports the usual layout with each series in one directory.
This viewer is for anatomical-coverage checks, not clinical diagnosis.
"""

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import SimpleITK as sitk


def read_cts(path: Path):
    """Read one volume file, or discover separate CT series recursively."""
    if path.is_file():
        yield sitk.ReadImage(str(path), sitk.sitkFloat32)
        return

    for directory, subdirs, filenames in os.walk(path):
        subdirs.sort()
        if not filenames:
            continue
        for uid in sitk.ImageSeriesReader.GetGDCMSeriesIDs(directory) or ():
            try:
                files = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(directory, uid)
                header = sitk.ImageFileReader()
                header.SetFileName(files[0])
                header.ReadImageInformation()
                if not header.HasMetaDataKey("0008|0060"):
                    continue
                if header.GetMetaData("0008|0060").strip().upper() != "CT":
                    continue
                print(f"Reading CT series: {len(files)} files in {directory}", flush=True)
                reader = sitk.ImageSeriesReader()
                reader.SetFileNames(files)  # GDCM orders by scan position, not filename.
                reader.SetOutputPixelType(sitk.sitkFloat32)
                image = reader.Execute()
            except RuntimeError as exc:
                print(f"WARNING: skipped unreadable series in {directory}: {exc}")
                continue
            yield image


def browse(image: sitk.Image, series: int, output: Path) -> bool:
    """Return True to continue to the next series, False to quit."""
    if image.GetDimension() != 3 or image.GetNumberOfComponentsPerPixel() != 1:
        print(f"Skipping series {series}: expected a scalar 3D CT volume.")
        return True

    # Reorder/flip to LPS; no cropping or intensity normalization.
    # Oblique scans retain their obliquity; this is not an axial resampling tool.
    image = sitk.DICOMOrient(image, "LPS")
    volume = sitk.GetArrayFromImage(image)  # [z, y, x]
    sx, sy, sz = image.GetSpacing()
    count = volume.shape[0]
    pages = (count + 19) // 20
    print(f"Series {series}: {count} slices; spacing = {sx:.3f}, {sy:.3f}, {sz:.3f} mm")
    print(f"Intensity range: {volume.min():.1f} to {volume.max():.1f}")
    if volume.min() >= 0 and volume.max() <= 1:
        print("WARNING: this looks normalized or like a mask. Use the original CT in HU.")

    state = {"page": 0, "lung": True, "continue": False}
    fig, axes = plt.subplots(4, 5, figsize=(16, 11))
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.05, top=0.91,
                        wspace=0.03, hspace=0.18)
    fig.text(0.5, 0.015,
             "Any other key: next | Left/B: previous | W: window | S: save PNG | Q/Esc: quit",
             ha="center", fontsize=10)

    def draw():
        start = state["page"] * 20
        name = "lung" if state["lung"] else "soft tissue"
        low, high = (-1350, 150) if state["lung"] else (-160, 240)
        for offset, ax in enumerate(axes.flat):
            ax.clear()
            ax.set_axis_off()
            z = start + offset
            if z < count:
                ax.imshow(volume[z], cmap="gray", vmin=low, vmax=high,
                          origin="upper", aspect=sy / sx, interpolation="nearest")
                ax.set_title(f"z = {z}", fontsize=9, pad=3)
        fig.suptitle(
            f"CT series {series:02d} | Page {state['page'] + 1}/{pages} | "
            f"Slices {start}-{min(start + 19, count - 1)} / {count} total\n"
            f"{name} window | Slice indices are zero-based after LPS reorientation",
            fontsize=12)
        fig.canvas.draw_idle()

    def on_key(event):
        key = (event.key or "").lower()
        if key in ("q", "escape"):
            plt.close(fig)
        elif key == "s":
            output.mkdir(parents=True, exist_ok=True)
            window = "lung" if state["lung"] else "soft"
            filename = output / f"series_{series:02d}_page_{state['page'] + 1:03d}_{window}.png"
            fig.savefig(filename, dpi=150)
            print(f"Saved: {filename.resolve()}")
        elif key == "w":
            state["lung"] = not state["lung"]
            draw()
        elif key in ("left", "b", "backspace"):
            state["page"] = max(0, state["page"] - 1)
            draw()
        elif key and key not in ("shift", "control", "ctrl", "alt", "super"):
            if state["page"] + 1 < pages:
                state["page"] += 1
                draw()
            else:
                state["continue"] = True
                plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", on_key)
    draw()
    plt.show(block=True)
    return state["continue"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="CT volume (.nii/.nii.gz) or patient DICOM folder")
    parser.add_argument("--output", type=Path, default=Path("ct_pages"), help="Saved PNG folder")
    args = parser.parse_args()
    if not args.input.exists():
        parser.error(f"Input does not exist: {args.input}")
    backend = plt.get_backend().lower()
    if backend in ("agg", "pdf", "svg", "ps", "cairo", "template") or "inline" in backend:
        parser.error("A graphical display is required. Run in a desktop terminal, not an inline notebook.")
    # Prevent Matplotlib's standard shortcuts from conflicting with ours.
    for setting in list(plt.rcParams):
        if setting.startswith("keymap."):
            plt.rcParams[setting] = []
    found = False
    for series, image in enumerate(read_cts(args.input), start=1):
        found = True
        if not browse(image, series, args.output):
            break
    if not found:
        parser.error("No readable CT DICOM series found. For NIfTI, pass the CT file itself.")


if __name__ == "__main__":
    main()
