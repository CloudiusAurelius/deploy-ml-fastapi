import pandas as pd
import argparse
import logging


def clean_column_names(input_file, output_file=None):
    # Load CSV
    logger.info(f"Loading CSV file: {input_file}")
    df = pd.read_csv(input_file)

    # Clean column names
    logger.info("Cleaning column names")
    df.columns = df.columns\
                   .str.strip()\
                   .str.lower()\
                   .str.replace(" ", "_")

    # If no output file is given, overwrite the input file
    if output_file is None:
        output_file = input_file

    # Save cleaned file
    logger.info(f"Saving cleaned CSV to: {output_file}")
    df.to_csv(output_file, index=False)
    print(f"Cleaned CSV saved to: {output_file}")


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Clean column names in a CSV file.")
    parser.add_argument("--input_file", "-i", help="Path to the input CSV file")
    parser.add_argument("--output_file", "-o", help="Path to save the cleaned CSV (default: overwrite input)")

    args = parser.parse_args()
    

    # Set up logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
    logger = logging.getLogger()

    logger.info("Starting column name cleaning script")

    # Clean the column names in the input file and save to output file
    clean_column_names(args.input_file, args.output_file)

