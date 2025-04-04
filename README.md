# 🍕 Hackapizza Community Edition 🚀

## Overview

This project was developed for the **Hackapizza Kaggle Competition** by **Stefano Iannicelli** and **Ettore Caputo**.  
The goal? To create a super-smart solution for **context-aware question answering** using **structured menu data** .  
We focused on **efficient token usage**, **precise data extraction**, and **multi-phase query processing** .

## 🔑 Key Features

- **📂 Structured Data Processing:**  
  - Splits menu files into structured chunks like headers and dishes 
  - Tags dish names with `<dish></dish>` for easy spotting 
  - Rebuilds tables and decodes Roman numerals using regex 

- **🧠 Multi-Expert System:**  
  - **Tech Expert**: Understands fancy cooking methods from the *Galactic Code* 
  - **Distance Expert**: Finds restaurants by planetary distances 
  - **Menu Header Expert**: Filters based on restaurant metadata 
  - **Menu Corpus Expert**: Dives deep into the menu content for dish details 

- **🔎 Boolean Query Processing:**  
  - Transforms user queries into **boolean expressions**  
  - Filters menu data with structured keyword logic   
  - Ensures **super precise answers** every time   

- **⚙️ Token Efficiency:**  
  - Minimizes dependence on LLMs thanks to boolean smarts   
  - Makes every token count for **context-aware replies** 

## 🏗️ Architecture

1. **🔑 Keyword Extraction** – Pulls out the important bits from the question  
2. **🛠️ Query Reformulation** – Turns them into boolean expressions  
3. **🧠 Expert Activation** – Different experts handle their part of the query  
4. **📚 Boolean Search** – Finds the matching data  
5. **🍽️ Final Answer Extraction** – Grabs dish names straight from the filtered content  

## 📊 Results

| 🧪 Configuration        | 🎯 Score (%) |
|------------------------|-------------|
| **Menu Expert Only**   | 63.5        |
| **+ Distance Expert**  | 66.7        |
| **+ Tech Expert**      | 76.5        |

## 🧩 Challenges & 🚀 Future Improvements

- **📐 Rigid Boolean Model** – Very structured queries; even small keyword slips can cause issues 
- **🧠 Tech Expert Optimization** – Currently sends the whole *Galactic Code* to the LLM. Switching to chunk-based retrieval could save tons of tokens!

---

📄 Want more details? Check out the **project PDF**! Thanks for reading! 🙌
