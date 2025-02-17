#!/usr/bin/env python3
"""
Combine pathology and consultation note text files into a single merged file per patient.

Usage:
  python combine_path_cons.py --input_dir /path/to/HNC_Reports \
                              --output_dir /path/to/PathConsCombined
"""

import os
import argparse
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def combine_reports(input_dir, output_dir):
    """
    Combines PathologyReports/*.txt and ConsultRedacted/*.txt for each patient into one text file.
    The combined files are saved under output_dir with the same base filename (patient ID).
    """
    pathology_folder = os.path.join(input_dir, "PathologyReports")
    consultation_folder = os.path.join(input_dir, "ConsultRedacted")

    # Gather patient IDs from both folders
    path_ids = set()
    cons_ids = set()

    if os.path.isdir(pathology_folder):
        for f in os.listdir(pathology_folder):
            if f.endswith(".txt"):
                path_ids.add(os.path.splitext(f)[0])

    if os.path.isdir(consultation_folder):
        for f in os.listdir(consultation_folder):
            if f.endswith(".txt"):
                cons_ids.add(os.path.splitext(f)[0])

    all_ids = sorted(path_ids.union(cons_ids))
    logger.info(f"Found {len(all_ids)} total patient IDs from path+cons.")

    os.makedirs(output_dir, exist_ok=True)

    for pid in all_ids:
        path_file = os.path.join(pathology_folder, pid + ".txt")
        cons_file = os.path.join(consultation_folder, pid + ".txt")

        combined_text = ""
        if os.path.isfile(path_file):
            with open(path_file, 'r', encoding='utf-8') as f:
                pathology_text = f.read()
            combined_text += pathology_text.strip() + "\n\n"

        if os.path.isfile(cons_file):
            with open(cons_file, 'r', encoding='utf-8') as f:
                consultation_text = f.read()
            combined_text += consultation_text.strip()

        if not combined_text.strip():
            logger.warning(f"No content for patient {pid}, skipping.")
            continue

        out_path = os.path.join(output_dir, pid + ".txt")
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(combined_text)

        logger.info(f"Combined text saved to {out_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Combine pathology and consultation note text files into one file per patient."
    )
    parser.add_argument("--input_dir", required=True,
                        help="Directory containing 'PathologyReports/' and 'ConsultRedacted/'.")
    parser.add_argument("--output_dir", required=True,
                        help="Output directory to save combined text files.")
    args = parser.parse_args()

    combine_reports(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()


# python combine_path_cons.py \
#     --input_dir "/media/yujing/One Touch3/HNC_Reports" \
#     --output_dir "/media/yujing/One Touch3/HNC_Reports/PathConsCombined"
