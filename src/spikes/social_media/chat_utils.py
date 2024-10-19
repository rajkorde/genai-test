def write_to_file(strings: list[str], file_path: str) -> None:
    try:
        with open(file_path, "w") as file:
            for string in strings:
                file.write(string + "\n\n")
    except OSError as e:
        print(f"An error occurred while writing to the file: {e}")
