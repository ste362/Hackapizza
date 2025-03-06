import json
import os
import re

from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document


class BooleanQueryParser:
    def __init__(self, query):
        self.tokens = self.tokenize(query)
        self.pos = 0

    def tokenize(self, query):

        pattern = r'\(|\)|NOT|AND|OR|"[^"]+"|\w+'
        tokens = re.findall(pattern, query)
        return [t.upper() if t in {"AND", "OR", "NOT"} else t.replace('"',"").strip() for t in tokens]


    def parse(self):
        return self.expr()

    def expr(self):
        node = self.term()
        while self.pos < len(self.tokens) and self.tokens[self.pos] == "OR":
            self.pos += 1
            node = ("OR", node, self.term())
        return node

    def term(self):
        node = self.factor()
        while self.pos < len(self.tokens) and self.tokens[self.pos] == "AND":
            self.pos += 1
            node = ("AND", node, self.factor())
        return node

    def factor(self):
        if self.pos < len(self.tokens) and self.tokens[self.pos] == "NOT":
            self.pos += 1
            return ("NOT", self.factor())
        return self.primary()

    def primary(self):
        if self.pos < len(self.tokens) and self.tokens[self.pos] == "(":
            self.pos += 1
            node = self.expr()
            if self.pos < len(self.tokens) and self.tokens[self.pos] == ")":
                self.pos += 1
                return node
            raise ValueError("Unmatched parenthesis")
        elif self.pos < len(self.tokens):
            node = self.tokens[self.pos]
            self.pos += 1
            return node
        raise ValueError("Unexpected end of expression")



# # Esempio di utilizzo:
#query = ("Testa di idra AND (True OR NOT (True AND False))")
#booleanQuery = BooleanQueryParser(query).tokens
#print(booleanQuery)
# parser = BooleanQueryParser(query)
# parsed_tree = parser.parse()






def boolean_searcher(query,doc_splits, lower=True,header=False):

    def evaluate(tree):
        if isinstance(tree, tuple):
            if tree[0] == "AND":
                return evaluate(tree[1]) and evaluate(tree[2])
            elif tree[0] == "OR":
                return evaluate(tree[1]) or evaluate(tree[2])
            elif tree[0] == "NOT":
                return not evaluate(tree[1])

        if lower:
            return (tree.lower() in chunk.replace("\n", " ").lower())

        return (tree in chunk.replace("\n", " "))

    results = []
    if header:
        pre_id=-1
        for doc in doc_splits:
            chunk= doc.page_content
            if doc.metadata["id"]!=pre_id:
                pre_id = doc.metadata["id"]
                if evaluate(query):
                    results.append(doc)

    else:
        for doc in doc_splits:
            chunk= doc.page_content
            if evaluate(query):
                results.append(doc)

    dict_results = {}
    for doc in results:
        if doc.metadata["id"] not in dict_results:
            dict_results[doc.metadata["id"]] = [doc.page_content]

        dict_results[doc.metadata["id"]].append(doc.page_content)
    return dict_results

#
# query = """("Shard di Materia Oscura" AND "Uova di Fenice")"""
#
# menu_paths = [
#     os.path.join("Menu", path) for path in os.listdir("Menu") if path.endswith(".txt")
# ]
#
# # Load menu documents
# menu_docs = [TextLoader(path).load() for path in menu_paths]
# menu_docs_list = [item for sublist in menu_docs for item in sublist]
#
# doc_splits = my_splitter(menu_docs_list)
#
# parser = BooleanQueryParser(query)
# parsed_tree = parser.parse()
# print(parsed_tree)
#
# documents=boolean_searcher(parsed_tree,doc_splits)
#
# for doc in documents:
#     print(doc.metadata)
#     print(doc.page_content)
#     print("\n##############################################################################################\n")