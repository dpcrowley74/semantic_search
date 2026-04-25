# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "google-genai",
#     "pypdf",
# ]
# ///

from dataclasses import dataclass, field
from pathlib import Path
from gemini_model import GeminiModel
from gemini_query import QueryFile


@dataclass
class Document:
    path: Path
    points:list[dict] = field(default_factory=list)
    model: GeminiModel = GeminiModel()

    def extract_key_point(self) -> "Document":
        query_string= "Extract the most important point from the file"
        if len(self.points)>0:
            existing_points = [point["Point"] for point in self.points]
            query_string += "that is not" + " or ".join(existing_points)
        query = QueryFile(str(self.path),query_string)
        point = self.model.query_with_document(query)
        embedding = self.model.generate_embedding(point)
        self.points.append({"Point":point, "Embedding":embedding})
        return self

    def extract_key_points(self) -> "Document":
        while len(self.points)<10:
            self.extract_key_point()
        return self



    @classmethod
    def from_path_string(cls, path_string: str) ->"Document":
        path = Path(path_string)
        return cls(path).extract_key_points()

    def __str__(self):
        class_string = ""
        class_string += f"Document can be found at {self.path} with output: \n\n"
        page_count = 1
        for point in self.points:
            class_string += f"Page {page_count}: \n\n"
            class_string += point["Point"]
            class_string+= "\n\n"
            class_string+= f"Embedding equals: {point["Embedding"]}"
            page_count += 1
        return class_string
       

if __name__ == "__main__":
    import os
    absolute_path = os.path.abspath("../../data")
    files = os.listdir("../../data")

    for file in files:
        document = Document.from_path_string(os.path.join(absolute_path, file))
    print(document)

