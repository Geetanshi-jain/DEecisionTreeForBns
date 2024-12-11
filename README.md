# DEecisionTreeForBns


# 🚔 IPC Section Classifier 📜

The **IPC Section Classifier** is a Python-based project designed to analyze crime reports and automatically identify the relevant sections of the Indian Penal Code (IPC). Using Natural Language Processing (NLP), TF-IDF vectorization, and cosine similarity, this tool maps unstructured crime report texts to IPC sections with confidence levels using fuzzy logic.

## 🔍 Features

- **Automated Section Matching**: Classifies crime reports against predefined IPC sections based on their descriptions.
- **Natural Language Processing**: Processes and cleans input text for better accuracy.
- **Similarity Analysis**: Calculates similarity scores using TF-IDF and cosine similarity.
- **Fuzzy Confidence Levels**: Determines the confidence of section classification as High, Medium, or Low.
- **Customizable Knowledge Base**: Easily extend or modify the IPC sections and conditions.

## 🛠️ Technologies Used

- **Python** 🐍
- **scikit-learn** 🤖
- **Natural Language Toolkit (NLTK)** 🖍️
- **NumPy** 🔢
- **pandas** 📊

## 🚀 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download necessary NLTK data:
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   ```

## 🖍️ Usage

1. Prepare a crime report as a string input.
2. Run the script `ipc_section_classifier.py`.
3. View the output as a sorted table showing:
   - IPC Section
   - Title
   - Confidence Level (High/Medium/Low)
   - Similarity Score

### Example

```python
crime_report = "An individual was found unlawfully discharging firearms in a public place."
```

### Output

| Section      | Title                        | Confidence Level | Similarity Score |
|--------------|------------------------------|------------------|------------------|
| Section 16   | Unlawful Use of Weapons      | High             | 0.85             |
| Section 14   | Possession of Firearms       | Medium           | 0.65             |

## 📂 File Structure

```
🗁 IPC Section Classifier  
│  
├── ipc_section_classifier.py    # Main Python script  
├── knowledge_base.json          # JSON file defining IPC sections and conditions  
├── requirements.txt             # Python dependencies  
└── README.md                    # Project documentation
```

## 🔮 Future Enhancements

- 📊 **Interactive Dashboard**: Add a front-end interface for better user interaction.
- 🌐 **API Integration**: Provide REST APIs for real-time classification.
- 📚 **Enhanced Knowledge Base**: Include more IPC sections for wider applicability.
- 🧐 **ML-based Model**: Use supervised learning for improved classification.



