from google import genai
from gemini_query import QueryFile

class GeminiModel:
    client = genai.Client()

    def query_with_document(self, file_query:QueryFile):
        uploaded_file = self.client.files.upload(file=file_query.file_path)
        try:
            response = self.client.models.generate_content(
                    model = 'gemini-2.5-flash',
                    contents = [
                        uploaded_file,
                        "\n\n",
                        file_query.query
                        ]
                    )
            return response.text
        except Exception as e:
            print (f"An Error occurred: {e}")
    
    def generate_embedding(self, contents: str):
        try:
            result = self.client.models.embed_content(
                model = "gemini-embedding-001",
                contents=contents,
                )
            return result.embeddings
        except Exception as e:
            print(f"An Error occurred: {e}")
            return e

