import pandas as pd
import argparse
import os

def clean_column_names(input_file, output_file=None):
    # Load CSV
    df = pd.read_csv(input_file)

    # Clean column names
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # If no output file is given, overwrite the input file
    if output_file is None:
        output_file = input_file

    # Save cleaned file
    df.to_csv(output_file, index=False)
    print(f"Cleaned CSV saved to: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean column names in a CSV file.")
    parser.add_argument("input_file", help="Path to the input CSV file")
    parser.add_argument("--output", "-o", help="Path to save the cleaned CSV (default: overwrite input)")

    args = parser.parse_args()
    clean_column_names(args.input_file, args.output)

