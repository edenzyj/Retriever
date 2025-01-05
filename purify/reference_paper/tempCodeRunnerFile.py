import os

def read_lines_from_txt_files(directory):
    """
    Reads all lines from .txt files in the specified directory, stripping whitespace from each line.

    Args:
        directory (str): Path to the directory containing .txt files.

    Returns:
        dict: A dictionary where keys are filenames and values are the count of stripped lines.
    """
    txt_line_counts = {}

    # Iterate through all files in the directory
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):  # Check if the file has a .txt extension
            file_path = os.path.join(directory, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                # Count lines after stripping whitespace
                txt_line_counts[filename] = sum(1 for line in file if line.strip())

    return txt_line_counts

# Example usage:
directory_path = "./"
all_txt_line_counts = read_lines_from_txt_files(directory_path)

# all_txt_line_counts now contains the line counts for each file.
