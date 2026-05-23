"""
Shared configuration and resources for graph nodes.
"""

import json
import os
import re
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document

# Initialize LLM
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=0,
    api_key="KEY",
)

# Load data paths
menu_paths = [
    os.path.join("Hackapizza Dataset/Menu_Final_txt", path) for path in os.listdir("Hackapizza Dataset/Menu_Final_txt") if path.endswith(".txt")
]
technical_paths = [
    "Hackapizza Dataset/Techs/Codice Galattico.txt",
]

# Load distance file
dist_file = open("Hackapizza Dataset/Techs/distanze.txt", "r").read()

# Load dish mapping
dish_mapping_path = "Hackapizza Dataset/Misc/dish_mapping.json"
with open(dish_mapping_path, "r") as f:
    dish_mapping = json.load(f)

# Load documents with UTF-8 encoding
menu_docs = [TextLoader(path, encoding='utf-8').load() for path in menu_paths]
menu_docs_list = [item for sublist in menu_docs for item in sublist]
tech_docs = [TextLoader(path, encoding='utf-8').load() for path in technical_paths]
tech_docs_list = [item for sublist in tech_docs for item in sublist]

# Text splitter configuration
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=200, chunk_overlap=20
)

# Helper functions
def roman_to_int(roman: str) -> int:
    """Convert Roman numeral to integer."""
    roman_values = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
    total = 0
    prev_value = 0

    for char in reversed(roman):
        value = roman_values.get(char, 0)
        if value < prev_value:
            total -= value
        else:
            total += value
        prev_value = value

    return total

def replace_roman_numerals(doc: Document) -> str:
    """Replace Roman numerals in document with integers."""
    text = doc.page_content

    def replacement(match):
        roman_numeral = match.group()
        if roman_numeral == "I" and (match.start() == 0 or text[match.start() - 2] in {'.', '!', '?'}):
            return roman_numeral
        return str(roman_to_int(roman_numeral))

    pattern = r"(?<![\w'])(M{0,4}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3}))(?![\w'])"
    return re.sub(pattern, replacement, text)

def dish_labeler(text: str):
    """Label dish names with tags."""
    for dish in sorted(dish_mapping.keys(), key=lambda k: len(k), reverse=True):
        text = text.replace(
            dish, f"<dish>{dish}</dish>", -1
        )
    return text

def dish_labeler_doc_with_underscore(doc):
    """Label dish names in document with underscores."""
    for dish in sorted(dish_mapping.keys(), key=lambda k: len(k), reverse=True):
        doc.page_content = doc.page_content.replace(
            dish, f"<dish>{dish.replace(" ", "_")}</dish>", -1
        )
    return doc

def my_splitter(menu_docs_list):
    """Custom document splitter based on dish keywords."""
    doc_splits = []
    for id, doc in enumerate(menu_docs_list):
        list_pos = []
        for keyword in dish_mapping.keys():
            pos = doc.page_content.find(keyword)
            if pos != -1:
                list_pos.append((keyword, pos))

        list_pos = sorted(list_pos, key=lambda x: x[1])
        start = 0
        for _, end in list_pos:
            if start < end:
                doc_splits.append(Document(metadata={"id": id}, page_content=doc.page_content[start:end]))
                start = end
        doc_splits.append(Document(metadata={"id": id}, page_content=doc.page_content[start:]))

    return doc_splits

# Process documents
doc_splits = my_splitter(menu_docs_list)
tech_doc_splits = text_splitter.split_documents(tech_docs_list)

# Validate splits
assert all([len(doc.page_content) > 0 for doc in doc_splits])
assert all([len(doc.page_content) > 0 for doc in tech_doc_splits])

# Label dish names
doc_splits = [dish_labeler_doc_with_underscore(doc) for doc in doc_splits]
tech_doc_splits = [dish_labeler_doc_with_underscore(doc) for doc in tech_doc_splits]