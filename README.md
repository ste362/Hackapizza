# Hackapizza Community Edition

## Overview

It is a project developed for the **Hackapizza Kaggle Competition** by Stefano Iannicelli and Ettore Caputo. The goal was to optimize **context-aware question answering** using **structured menu data**. The solution focuses on **efficient token usage, precise data extraction, and multi-phase query processing**.

## Key Features

- **Structured Data Processing:**  
  - Splitting menu files into structured chunks (header, individual dishes).  
  - Tagging dish names with `<dish></dish>` for easier extraction.  
  - Rebuilding tables and converting Roman numerals using regex.

- **Multi-Expert System:**  
  - **Tech Expert**: Extracts cooking techniques from the "Galactic Code".  
  - **Distance Expert**: Retrieves restaurants based on planetary distances.  
  - **Menu Header Expert**: Filters results using restaurant metadata.  
  - **Menu Corpus Expert**: Extracts dish information from menu content.  

- **Boolean Query Processing:**  
  - Converts user queries into **boolean expressions**.  
  - Searches menu data using structured keyword filtering.  
  - Ensures **high precision** in response generation.  

- **Token Efficiency:**  
  - Reduces reliance on LLMs by applying boolean retrieval techniques.  
  - Optimizes token usage in context-aware responses.  

## Architecture

1. **Keyword Extraction:** Identifies relevant terms from user queries.
2. **Query Reformulation:** Constructs boolean expressions based on keywords.
3. **Expert Activation:** Each expert processes relevant parts of the query.
4. **Boolean Search:** Extracts menu data based on structured queries.
5. **Final Answer Extraction:** Identifies dish names directly from the filtered text.

## Results

| Configuration      | Score (%) |
|-------------------|----------|
| **Menu Expert Only** | 63.5 |
| **+ Distance Expert** | 66.7 |
| **+ Tech Expert** | 76.5 |

## Challenges & Future Improvements

- **Rigid Boolean Model:** Queries are highly structured; minor keyword errors can lead to failed retrievals.
- **Tech Expert Optimization:** Currently, the entire "Galactic Code" is passed to the LLM. A chunk-based retrieval system could significantly improve token efficiency.

## More info in the pdf file
