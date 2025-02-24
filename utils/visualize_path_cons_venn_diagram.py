#!/usr/bin/env python3

import os
import argparse
import logging

import matplotlib.pyplot as plt
from matplotlib_venn import venn2

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def gather_ids(folder_path: str) -> set:
    """Return set of patient IDs (filename minus extension) for .txt files in a folder."""
    s = set()
    if not os.path.isdir(folder_path):
        logger.warning(f"Folder not found: {folder_path}")
        return s
    for fn in os.listdir(folder_path):
        if fn.endswith(".txt"):
            pid = os.path.splitext(fn)[0]
            s.add(pid)
    return s

def main():
    parser = argparse.ArgumentParser(
        description="Generate a 2-set Venn diagram for pathology vs. consultation patients, shifting the set labels."
    )
    parser.add_argument("--input_dir", required=True,
                        help="Parent directory with PathologyReports/ and ConsultRedacted/ subfolders.")
    parser.add_argument("--output_file", default="venn_diagram.png",
                        help="Path to save the venn diagram image.")
    args = parser.parse_args()

    path_folder = os.path.join(args.input_dir, "PathologyReports")
    cons_folder = os.path.join(args.input_dir, "ConsultRedacted")

    path_ids = gather_ids(path_folder)
    cons_ids = gather_ids(cons_folder)

    logger.info(f"Pathology count: {len(path_ids)}")
    logger.info(f"Consultation count: {len(cons_ids)}")
    logger.info(f"Overlap: {len(path_ids.intersection(cons_ids))}")

    plt.figure(figsize=(6,6))

    # Create the venn
    venn = venn2(
        [path_ids, cons_ids],
        set_labels=("Pathology \nReports", "Consultation \nNotes")
    )

    # Grab the set labels (IDs 'A' and 'B' in venn2)
    lblA = venn.get_label_by_id('A')  # left circle label
    lblB = venn.get_label_by_id('B')  # right circle label

    # Center-align each multiline label
    if lblA is not None:
        lblA.set_fontsize(18)
        lblA.set_ha('center')
        lblA.set_multialignment('center')
    if lblB is not None:
        lblB.set_fontsize(18)
        lblB.set_ha('center')
        lblB.set_multialignment('center')

    # Shift them horizontally (and optionally vertically) to avoid overlap
    if lblA is not None:
        current_pos_A = lblA.get_position()  # (x, y)
        # Move "Pathology Reports" label more left and slightly downward
        lblA.set_position((current_pos_A[0] - 0.2, current_pos_A[1] - 0.06))

    if lblB is not None:
        current_pos_B = lblB.get_position()
        # Move "Consultation Notes" label more right and slightly downward
        lblB.set_position((current_pos_B[0] + 0.2, current_pos_B[1] - 0.0001))

    # Optionally set custom colors
    def rgb_norm(r,g,b):
        return (r/255.0, g/255.0, b/255.0)
    color_path_only = rgb_norm(114,155,87)
    color_cons_only = rgb_norm(142,93,176)
    color_overlap   = rgb_norm(187,95,76)

    patch_10 = venn.get_patch_by_id('10')
    if patch_10 is not None:
        patch_10.set_color(color_path_only)

    patch_01 = venn.get_patch_by_id('01')
    if patch_01 is not None:
        patch_01.set_color(color_cons_only)

    patch_11 = venn.get_patch_by_id('11')
    if patch_11 is not None:
        patch_11.set_color(color_overlap)

    # Adjust the region-number labels
    subset_10 = venn.get_label_by_id('10')
    subset_01 = venn.get_label_by_id('01')
    subset_11 = venn.get_label_by_id('11')
    if subset_10 is not None:
        subset_10.set_fontsize(15)
    if subset_01 is not None:
        subset_01.set_fontsize(15)
    if subset_11 is not None:
        subset_11.set_fontsize(15)

    plt.savefig(args.output_file, dpi=600, bbox_inches='tight')
    logger.info(f"Venn diagram saved: {args.output_file}")

if __name__ == "__main__":
    main()


# Usage Example:
# python visualize_path_cons_venn_diagram.py \
#   --input_dir "/media/yujing/One Touch3/HNC_Reports" \
#   --output_file "/Data/Yujing/HNC_OutcomePred/Reports_Agents/Figures_Tables/Figure1_VennDiagram/path_cons_venn_v1.png"

