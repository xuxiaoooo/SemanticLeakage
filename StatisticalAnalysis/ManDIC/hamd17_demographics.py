#!/usr/bin/env python3
"""
Generate demographic tables for ManDIC using HAMD-17 only.
Only includes samples where both CSV record and audio file exist.
"""

from __future__ import annotations

import argparse
import wave
from pathlib import Path

import numpy as np
import pandas as pd

SEX_LABELS = {
    1: "Male",
    0: "Female",
}


def format_mean_sd(series: pd.Series, decimals: int = 2) -> str:
    series = series.dropna()
    if series.empty:
        return "NA"
    mean = series.mean()
    sd = series.std(ddof=1)
    return f"{mean:.{decimals}f} +/- {sd:.{decimals}f}"


def format_n_pct(count: int, total: int) -> str:
    pct = (count / total * 100.0) if total else 0.0
    return f"{count} ({pct:.1f}%)"


def render_markdown_table(rows: list[tuple[str, str]], header: tuple[str, str]) -> str:
    lines = [
        f"| {header[0]} | {header[1]} |",
        "| --- | --- |",
    ]
    for label, value in rows:
        lines.append(f"| {label} | {value} |")
    return "\n".join(lines)


def build_table1(df: pd.DataFrame) -> str:
    total = len(df)
    rows: list[tuple[str, str]] = [
        ("N", f"{total}"),
        ("Age, years (mean +/- SD)", format_mean_sd(df["age"])),
    ]

    sex_counts = df["sex"].value_counts(dropna=False)
    for sex_value, label in SEX_LABELS.items():
        count = int(sex_counts.get(sex_value, 0))
        rows.append((f"{label}, n (%)", format_n_pct(count, total)))

    known_counts = sum(int(sex_counts.get(k, 0)) for k in SEX_LABELS)
    unknown_count = total - known_counts
    if unknown_count:
        rows.append(("Sex, other/unknown, n (%)", format_n_pct(unknown_count, total)))

    rows.append(
        ("HAMD-17 total score (mean +/- SD)", format_mean_sd(df["HAMD-17_total_score"]))
    )
    return render_markdown_table(rows, header=("Characteristic", "Value"))


def build_severity_table(df: pd.DataFrame) -> str:
    bins = [-float("inf"), 7, 16, 23, float("inf")]
    labels = ["Normal (0-7)", "Mild (8-16)", "Moderate (17-23)", "Severe (>=24)"]
    severity = pd.cut(df["HAMD-17_total_score"], bins=bins, labels=labels)
    counts = severity.value_counts().reindex(labels).fillna(0).astype(int)

    total = len(df)
    rows = [(label, format_n_pct(int(counts[label]), total)) for label in labels]
    return render_markdown_table(rows, header=("HAMD-17 severity", "n (%)"))


def get_wav_duration(wav_path: Path) -> float:
    """Get duration of WAV file in seconds."""
    with wave.open(str(wav_path), 'rb') as wf:
        frames = wf.getnframes()
        rate = wf.getframerate()
        return frames / float(rate)


def get_valid_samples(root: Path) -> tuple[pd.DataFrame, list[dict]]:
    """Get samples where CSV record, audio file, and HAMD-17 score all exist."""
    data_path = root / "data" / "ManDIC" / "info.csv"
    audio_dir = root / "data" / "ManDIC" / "data"
    
    df = pd.read_csv(data_path)
    
    # Get existing audio files
    audio_files = {f.stem: f for f in audio_dir.glob("*.WAV")}
    
    # Filter: CSV + audio + HAMD-17 all exist
    df = df[
        df["standard_id"].isin(audio_files.keys()) & 
        df["HAMD-17_total_score"].notna()
    ].copy()
    
    # Get duration directly from WAV files
    durations = []
    valid_ids = []
    for sample_id in df["standard_id"]:
        wav_path = audio_files.get(sample_id)
        if wav_path:
            try:
                dur = get_wav_duration(wav_path)
                durations.append({"sample_id": sample_id, "duration_sec": dur})
                valid_ids.append(sample_id)
            except Exception:
                pass
    
    # Only keep samples with valid duration
    df = df[df["standard_id"].isin(valid_ids)].copy()
    
    return df, durations


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate demographic tables for ManDIC using HAMD-17 only."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "outputs",
        help="Directory to write markdown tables.",
    )
    parser.add_argument(
        "--with-severity-table",
        action="store_true",
        help="Also generate a HAMD-17 severity distribution table.",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    
    # Get valid samples (both CSV and audio exist)
    df, durations = get_valid_samples(root)
    
    print(f"\n{'='*60}")
    print("VALID SAMPLES (CSV + Audio + HAMD-17)")
    print(f"{'='*60}")
    print(f"Valid samples: {len(df)}")
    
    # Duration statistics
    if durations:
        dur_values = [d["duration_sec"] for d in durations]
        total_sec = sum(dur_values)
        print(f"\nDURATION STATISTICS:")
        print(f"  Total duration: {total_sec:.2f} sec = {total_sec/60:.2f} min = {total_sec/3600:.2f} hours")
        print(f"  Mean duration: {np.mean(dur_values):.2f} sec")
        print(f"  Median duration: {np.median(dur_values):.2f} sec")
        print(f"  Std duration: {np.std(dur_values):.2f} sec")
        print(f"  Min duration: {min(dur_values):.2f} sec")
        print(f"  Max duration: {max(dur_values):.2f} sec")
    print(f"{'='*60}\n")
    
    if df.empty:
        print("No valid samples found.")
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)

    table1 = build_table1(df)
    table1_path = args.output_dir / "table1_hamd17_demographics.md"
    table1_title = "Table 1. Demographic characteristics (HAMD-17 available participants)."
    table1_text = f"{table1_title}\n\n{table1}\n"
    table1_path.write_text(table1_text, encoding="utf-8")
    print(table1_text)
    print(f"Saved: {table1_path}")

    if args.with_severity_table:
        table2 = build_severity_table(df)
        table2_path = args.output_dir / "table2_hamd17_severity.md"
        table2_title = "Table 2. HAMD-17 severity distribution."
        table2_text = f"{table2_title}\n\n{table2}\n"
        table2_path.write_text(table2_text, encoding="utf-8")
        print(table2_text)
        print(f"Saved: {table2_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
