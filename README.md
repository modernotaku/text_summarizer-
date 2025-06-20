# text_summarizer-
A tool that summarizes long articles or documents into concise summaries


Installation: 
bash# 

# Core dependencies
pip install transformers torch

# Optional for file support
pip install PyPDF2 python-docx


Command Line:
bash# 
# Summarize a single file
python text_summarizer.py article.txt

# Custom parameters
python text_summarizer.py document.pdf --max-length 200 --output summary.txt

# Process entire directory
python text_summarizer.py ./documents/ --model facebook/bart-large-cnn


Python Library :
#Python
pythonfrom text_summarizer import TextSummarizer

summarizer = TextSummarizer()
result = summarizer.summarize_text("Your long text here...")
print(f"Summary: {result['summary']}")
print(f"Compressed {result['compression_ratio']}:1")
