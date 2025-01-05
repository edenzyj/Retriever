import os

def read_lines_from_txt_files(directory):
    """
    Reads all lines from .txt files in the specified directory, stripping whitespace from each line.

    Args:
        directory (str): Path to the directory containing .txt files.

    Returns:
        int: The total count of stripped lines across all .txt files.
    """
    total_line_count = 0

    # Iterate through all files in the directory
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):  # Check if the file has a .txt extension
            file_path = os.path.join(directory, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                # Count lines after stripping whitespace
                total_line_count += sum(1 for line in file if line.strip())

    return total_line_count

# Example usage:
directory_path = "./"
total_lines = read_lines_from_txt_files(directory_path)
print(total_lines)
# all_txt_line_counts now contains the line counts for each file.
