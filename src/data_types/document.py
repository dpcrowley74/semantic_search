# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "google-genai",
#     "langchain-text-splitters",
#     "pypdf",
# ]
# ///

from dataclasses import dataclass, field
from pathlib import Path
from pypdf import PdfReader
from gemini_model import GeminiModel
from gemini_query import QueryFile
from langchain_text_splitters import RecursiveCharacterTextSplitter

@dataclass
class Document:
    path: Path
    model: "GeminiModel"
    summary:str=""
    summary_embedding:list=field(default_factory=list)
    document_embedding:list =field(default_factory=list)

    def read_pdf(self)->str:
        reader=PdfReader(self.path)
        all_text = ""
        for page in reader.pages:
            text=page.extract_text()
            if text:
                all_text+= text + "\n"
        return all_text

    def embed_pdf(self)->"Document":
        text = self.read_pdf()
        self.document_embedding = self.embed_data(text)
        return self

    def construct_summary(self)-> "Document":
        query = QueryFile(str(self.path), 
                        "Extract the key points from the document"
                          )
        self.summary = self.model.query_with_document(query)
        self.summary_embedding = self.embed_data(self.summary)
        return self

    def embed_data(self, data)-> list:
        text_splitter= RecursiveCharacterTextSplitter(chunk_size = 100, chunk_overlap=20)
        texts=text_splitter.split_text(data)
        embedded_text=[]
        for text in texts:
            embedded_text.append({
                "Text": text,
                "Embedding":self.model.generate_embedding(text)
                })
        return embedded_text

    @classmethod
    def from_path_string(cls, path_string: str, model: "GeminiModel") ->"Document":
        path = Path(path_string)
        return cls(path, model).construct_summary().embed_pdf()

    def __str__(self):
        class_string = ""
        class_string += f"Document can be found at {self.path} with output: \n\n"
        class_string += self.summary
        if len(self.summary_embedding)>0:
            class_string += "\n\n The summary has an embedding"
        else:
            class_string += "\n\n The summary is not embedded"
        if len(self.document_embedding)>0:
            class_string += "\n\n The Document has an embedding"
        else:
            class_string += "\n\n The Document is not embedded"
        return class_string

if __name__ == "__main__":
    import os
    absolute_path = os.path.abspath("../../data")
    files = os.listdir("../../data")

    for file in files:
        document = Document.from_path_string(os.path.join(absolute_path,
                                                          file), GeminiModel())
    print(document)

