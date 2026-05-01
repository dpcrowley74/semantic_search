# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "google-genai",
#     "langchain-text-splitters",
#     "polars",
#     "pypdf",
# ]
# ///

import os
from dataclasses import dataclass, field
from pathlib import Path
from pypdf import PdfReader
import polars as pl
from src.models.gemini_model import GeminiModel
from src.models.tools import QueryFile
from langchain_text_splitters import RecursiveCharacterTextSplitter

@dataclass
class Document:
    path: Path
    model: "GeminiModel"
    summary:str=""
    summary_embedding:"pl.DataFrame"=field(default_factory=pl.DataFrame)
    document_embedding:"pl.DataFrame"=field(default_factory=pl.DataFrame)

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

    def embed_data(self, data)-> "pl.DataFrame":
        text_splitter= RecursiveCharacterTextSplitter(chunk_size = 100, chunk_overlap=20)
        texts=text_splitter.split_text(data)
        embedded_text=[]
        for text in texts:
            [embedding] = self.model.generate_embedding(text)
            if embedding.values:
                embedded_text.append({
                    "Path": str(self.path),
                    "Text": text,
                    "Embedding":embedding.values
                    })
        return pl.from_dicts(embedded_text, schema={"Path":pl.String,
                                                    "Text":pl.String,
                                                    "Embedding":pl.Array(pl.Float16,512)})

    @classmethod
    def from_path_string(cls, path_string: str, model: "GeminiModel") ->"Document":
        path = Path(path_string)
        return cls(path, model).construct_summary().embed_pdf()

    def __str__(self):
        class_string = ""
        class_string += f"Document can be found at {self.path} with output: \n\n"
        class_string += self.summary
        if len(self.summary_embedding)>0:
            class_string += "\n\nThe Summary has an embedding"
        else:
            class_string += "\n\nThe Summary is not embedded"
        if len(self.document_embedding)>0:
            class_string += "\n\nThe Document has an embedding"
        else:
            class_string += "\n\nThe Document is not embedded"
        return class_string

    def save_class(self, knowledge_graph_location: str, vector_database_location: str):
        self.document_embedding.write_parquet(os.path.join(vector_database_location,
                                                           self.path.name.split(".")[0] +
                                                           ".parquet"))
        if not os.path.exists(os.path.join(knowledge_graph_location, "vector_database")):
            os.mkdir(os.path.join(knowledge_graph_location, "vector_database"))
        self.summary_embedding.write_parquet(os.path.join(knowledge_graph_location,
        "vector_database", self.path.name.split(".")[0] + ".parquet"))
        with open(os.path.join(knowledge_graph_location,
                               self.path.name.split(".")[0] +".md"), "w", encoding="utf-8") as f:
            f.write(self.__str__())

if __name__ == "__main__":
    absolute_path = os.path.abspath("data")
    files = os.listdir("data")

    for file in files:
        document = Document.from_path_string(os.path.join(absolute_path,
                                                          file), GeminiModel())
        document.save_class("output/knowledge_graph", "output/vector_database")
    print(document)

