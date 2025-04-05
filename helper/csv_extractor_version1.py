import pandas as pd

def extract_column_to_list(csv_filepath: str, column_name: str) -> list[str]:
    """
    Extracts a specific column from a CSV file and returns it as a list of strings.

    Args:
        csv_filepath: The path to the CSV file.
        column_name: The name of the column to extract.

    Returns:
        A list of strings containing the data from the specified column.
        Returns an empty list if the file is not found or the column doesn't exist.
    """
    try:
        # Read the CSV file into a Pandas DataFrame
        df = pd.read_csv(csv_filepath)

        # Check if the specified column exists in the DataFrame
        if column_name not in df.columns:
            print(f"Error: Column '{column_name}' not found in the CSV file.")
            return []

        # Extract the specified column as a Pandas Series
        column_series = df[column_name]

        # Convert the Pandas Series to a list of strings
        string_list = column_series.astype(str).tolist()

        return string_list

    except FileNotFoundError:
        print(f"Error: CSV file not found at '{csv_filepath}'.")
        return []
    except Exception as e:
        print(f"An error occurred: {e}")
        return []

# Example usage:
if __name__ == "__main__":
    # Create a sample CSV file for demonstration
    data = {'ID': [1, 2, 3],
            'Sentence': ["This is the first sentence.",
                         "Another interesting sentence here.",
                         "The final sentence in this example."],
            'Category': ['A', 'B', 'A']}
    sample_df = pd.DataFrame(data)
    sample_df.to_csv('sample.csv', index=False)

    # Specify the path to your CSV file and the name of the column you want to extract
    file_path = 'sample.csv'
    target_column = 'Sentence'

    # Call the function to extract the column
    sentences_list = extract_column_to_list(file_path, target_column)

    # Print the resulting list of strings
    if sentences_list:
        print(f"Extracted sentences from column '{target_column}':")
        for sentence in sentences_list:
            print(sentence)