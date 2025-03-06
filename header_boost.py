import json
import os

from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI


dish_mapping_path = "Hackapizza Dataset/Misc/dish_mapping.json"
with open(dish_mapping_path, "r") as f:
    dish_mapping = json.load(f)

import re


def roman_to_int(roman):
    roman_numerals = {
        'I': 1, 'V': 5, 'X': 10, 'L': 50,
        'C': 100, 'D': 500, 'M': 1000
    }
    total = 0
    prev_value = 0
    for char in reversed(roman):
        value = roman_numerals.get(char, 0)
        if value < prev_value:
            total -= value
        else:
            total += value
        prev_value = value
    return total


def replace_roman_with_int(text):
    roman_pattern = r"(?<![\w'])(M{0,4}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3}))(?![\w'])"

    def replacer(match):
        roman_str = match.group().upper()
        if roman_str:  # Ensure it's a valid Roman numeral
            return str(roman_to_int(roman_str))
        return match.group()

    return re.sub(roman_pattern, replacer, text)


def my_splitter(menu_docs_list):
    doc_splits = []
    for id,doc in enumerate(menu_docs_list):
        list_pos = []
        for keyword in dish_mapping.keys():
            pos = doc.page_content.find(keyword)
            if pos != -1:
                list_pos.append((keyword, pos))

        list_pos = sorted(list_pos, key=lambda x: x[1])
        start = 0
        for _, end in list_pos:
            if start < end:
                doc_splits.append(Document(metadata={"id":id}, page_content=doc.page_content[start:end]))
                start = end
        doc_splits.append(Document(metadata={"id":id}, page_content=doc.page_content[start:]))

    return doc_splits




menu_paths = [
    os.path.join("Hackapizza Dataset/Menu_Final_txt", path) for path in os.listdir("Hackapizza Dataset/Menu_Final_txt") if path.endswith(".txt")
]



llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=0,
    api_key="",
    # base_url="...",
    # organization="...",
    # other params...
)

for doc in menu_paths:
    print("Processing", doc)
    with open(doc, "r") as f:
        header=""
        line=""
        while line!="Menu\n":
            line=f.readline()
            header += line

        response=llm.invoke(f"""Riassumi il seguente testo in:
                        Nome Ristorante: ...
                        Nome Pianeta: ....
                        Nome Chef: ....
                        Lista di licenze che ha ottenuto il ristorante:["licenza 1", "licenza 2"]
                        
                        Esempio:
                            Testo:"Benvenuti nel ristorante Gli Echi della Luce sul pianeta Terra. Lo chef Barbieri vi fara' vivere una seranta indimenticabile. Il ristorante ha le seguenti licenze: P di grado 3, LHF di grado 5 "
                                Nome Ristorante: Gli Echi della Luce
                                Nome Pianeta: Terra
                                Nome Chef: Barbieri
                                Lista di licenze che ha ottenuto il ristorante:["P 3", "LHF 5"]
                            
                        Testo:"{header}"
        """)


        header = replace_roman_with_int(response.content)
        corpus=f.read()
        file=header+corpus

    with open(doc, "w") as f:
        f.write(file)

