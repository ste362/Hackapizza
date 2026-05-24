import os
from typing_extensions import TypedDict
from typing import Annotated
import operator
from langgraph.graph import StateGraph
import csv

# Import all graph nodes from the graph_node package
from graph_node import (
    extract_keywords,
    ask_to_distance_expert,
    ask_to_menu_header_expert,
    ask_to_menu_corpus_expert,
    ask_to_tech_expert,
    generate,
)

# Import shared configuration and resources
from graph_node.config import dish_mapping


#===========================================================================
#                            GRAPH STATE
#===========================================================================


class GraphState(TypedDict):
    """
    Graph state is a dictionary that contains information we want to propagate to, and modify in, each graph node.
    """

    question: str  # User question
    generation: str  # LLM generation
    max_retries: int  # Max number of retries for answer generation
    answers: int  # Number of answers generated
    loop_step: Annotated[int, operator.add]
    menu_query: str  # Menu expert query
    tech_query: str  # Technical expert query
    misc_query: str  # Misc expert query
    menu_expert_answer: dict  # Menu expert answer
    keywords_extractor_answer: dict
    menu_header_answer: dict
    tech_expert_answer: dict
    planet_distance_answer: list






# Initialize the workflow graph
workflow = StateGraph(GraphState)

# Add all processing nodes to the workflow
workflow.add_node("keywords_extractor", extract_keywords)
workflow.add_node("distance_expert", ask_to_distance_expert)
workflow.add_node("menu_header_expert", ask_to_menu_header_expert)
workflow.add_node("menu_corpus_expert", ask_to_menu_corpus_expert)
workflow.add_node("tech_expert", ask_to_tech_expert)
workflow.add_node("answer_expert", generate)

# Define the execution flow between nodes
workflow.set_entry_point("keywords_extractor")
workflow.add_edge("keywords_extractor","tech_expert")
workflow.add_edge("tech_expert","distance_expert")
workflow.add_edge("distance_expert","menu_header_expert")
workflow.add_edge("menu_header_expert","menu_corpus_expert")
workflow.add_edge(["menu_corpus_expert"], "answer_expert")

graph = workflow.compile()
#============================= END OF LANGCHAIN GRAPH ==============================



# Save workflow graph visualization
png_graph = graph.get_graph().draw_mermaid_png()
with open("graph.png", "wb") as f:
    f.write(png_graph)
print(f"Graph saved as 'graph.png' in {os.getcwd()}")

# Load test queries from CSV file


queries = []
with open("Hackapizza Dataset/domande.csv", "r", encoding="utf-8") as f:
    reader = csv.reader(f)
    for row in reader:
        queries.append(row[0])

queries = queries[1:]  # Skip header row

# Initialize output file for results
answers = []
with open("answers.csv", "w") as f:
    writer = csv.writer(f, delimiter=",")
    writer.writerow(["row_id", "result"])

# Process each query through the workflow graph
error = 0
start = 0
for i, query in enumerate(queries[start:]):
    print("ANSWERING TO QUERY:\t", query)

    # Stream query through the graph
    inputs = {"question": query}
    for event in graph.stream(inputs, stream_mode="values"):

        # Extract result when generation is complete
        if "generation" in event.keys():
            print("$$$ FINAL ANSWER:\7", event["generation"])
            answers.append(event["generation"])
            print("\n\n\n")
            try:
                # Clean dish names and map to IDs
                dishes = event["generation"]["dishes"]
                dishes = [x.replace("_", " ").replace("<dish>", "").replace("</dish>", "") for x in dishes]

                # Convert dish names to mapped IDs
                real_dishes = []
                for dish in dishes:
                    if dish in dish_mapping:
                        real_dishes.append(dish_mapping.get(dish))

                # Format output: use "1" for no results, otherwise join IDs
                dishes_text = "1" if len(real_dishes) == 0 else ",".join(str(dish) for dish in real_dishes)

            except:
                error += 1
                dishes_text = "1"
                print("Error")

            # Write result to output file
            with open("answers.csv", "a") as f:
                f.write(f'{i+1+start},"{dishes_text}"\n')

print("Total error:", error)

