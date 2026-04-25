# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "google-genai",
# ]
# ///

import os
from google import genai
from dataclasses import dataclass

@dataclass
class QueryFile:
    file_path: str
    query: str


def query_gemini_with_document(file_query:QueryFile):
    """
    Uploads a document to the Gemini API and asks a question about it.
    """
    client = genai.Client()
    uploaded_file = client.files.upload(file=file_query.file_path)
    print(f'{uploaded_file=}')

    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=[
            uploaded_file,
            "\n\n",
            file_query.query
            ]
        )
        
      
    return response.text


if __name__ == "__main__":
    # Example usage:
    # Create a dummy file to test with if you don't have one
    sample_file = "sample_document.txt"
    if not os.path.exists(sample_file):
        with open(sample_file, "w") as f:
            f.write("The secret passcode is: 42. Do not share it with anyone.")
    
    prompt = "What is the secret passcode mentioned in the document?"
    
    try:
        print(query_gemini_with_document(QueryFile(sample_file, prompt)))

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        os.remove("sample_document.txt")
        
