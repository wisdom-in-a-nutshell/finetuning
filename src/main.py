from data_preparation import prepare_data_for_gemini, InvalidDataFormatError

def main():
    try:
        data = prepare_data_for_gemini("path/to/your/data.jsonl")
        # Process the data further...
    except InvalidDataFormatError as e:
        print(f"Error: {e}")
        # Handle the error (e.g., exit the program or ask for a different file)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()