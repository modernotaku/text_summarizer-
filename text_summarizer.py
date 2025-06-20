#!/usr/bin/env python3
"""
Text Summarizer Tool
A comprehensive tool for summarizing long articles and documents using Hugging Face Transformers.
"""

import os
import re
import argparse
from typing import List, Optional, Dict, Any
from pathlib import Path
import logging

try:
    from transformers import pipeline, AutoTokenizer
    import torch
except ImportError:
    print("Required packages not installed. Please run:")
    print("pip install transformers torch")
    exit(1)

# Optional imports for document processing
try:
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

try:
    from docx import Document
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False


class TextSummarizer:
    """Advanced text summarization tool using Hugging Face models."""
    
    def __init__(self, model_name: str = "facebook/bart-large-cnn", device: Optional[str] = None):
        """
        Initialize the summarizer with a pre-trained model.
        
        Args:
            model_name: Hugging Face model identifier
            device: Device to run on ('cpu', 'cuda', or None for auto-detection)
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Load the summarization pipeline
        self.logger.info(f"Loading model: {model_name}")
        try:
            self.summarizer = pipeline(
                "summarization",
                model=model_name,
                device=0 if self.device == "cuda" else -1,
                return_tensors=True
            )
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.logger.info("Model loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise
    
    def clean_text(self, text: str) -> str:
        """Clean and preprocess text for summarization."""
        # Remove extra whitespace and newlines
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters that might interfere
        text = re.sub(r'[^\w\s.,!?;:()"-]', '', text)
        return text.strip()
    
    def chunk_text(self, text: str, max_chunk_length: int = 1000) -> List[str]:
        """
        Split text into chunks that fit within model constraints.
        
        Args:
            text: Input text to chunk
            max_chunk_length: Maximum tokens per chunk
            
        Returns:
            List of text chunks
        """
        # Tokenize to check length
        tokens = self.tokenizer.encode(text, truncation=False)
        
        if len(tokens) <= max_chunk_length:
            return [text]
        
        # Split into sentences for better chunking
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_tokens = len(self.tokenizer.encode(sentence))
            
            if current_length + sentence_tokens > max_chunk_length:
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                    current_chunk = [sentence]
                    current_length = sentence_tokens
                else:
                    # Sentence is too long, truncate it
                    truncated = self.tokenizer.decode(
                        self.tokenizer.encode(sentence)[:max_chunk_length]
                    )
                    chunks.append(truncated)
            else:
                current_chunk.append(sentence)
                current_length += sentence_tokens
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def summarize_text(self, text: str, max_length: int = 150, min_length: int = 50, 
                      chunk_size: int = 1000) -> Dict[str, Any]:
        """
        Summarize input text with comprehensive options.
        
        Args:
            text: Input text to summarize
            max_length: Maximum length of summary
            min_length: Minimum length of summary
            chunk_size: Maximum tokens per chunk for long texts
            
        Returns:
            Dictionary containing summary and metadata
        """
        if not text.strip():
            return {"summary": "", "chunks_processed": 0, "original_length": 0}
        
        # Clean the text
        cleaned_text = self.clean_text(text)
        original_length = len(cleaned_text.split())
        
        # Check if text needs chunking
        chunks = self.chunk_text(cleaned_text, chunk_size)
        
        if len(chunks) == 1:
            # Single chunk processing
            try:
                result = self.summarizer(
                    cleaned_text,
                    max_length=max_length,
                    min_length=min_length,
                    do_sample=False
                )
                summary = result[0]['summary_text']
            except Exception as e:
                self.logger.error(f"Summarization failed: {e}")
                summary = "Error: Could not generate summary"
        else:
            # Multi-chunk processing
            chunk_summaries = []
            
            for i, chunk in enumerate(chunks):
                try:
                    result = self.summarizer(
                        chunk,
                        max_length=min(max_length // len(chunks) + 50, 200),
                        min_length=min(min_length // len(chunks), 30),
                        do_sample=False
                    )
                    chunk_summaries.append(result[0]['summary_text'])
                    self.logger.info(f"Processed chunk {i+1}/{len(chunks)}")
                except Exception as e:
                    self.logger.error(f"Failed to process chunk {i+1}: {e}")
                    continue
            
            # Combine chunk summaries
            if chunk_summaries:
                combined_summary = ' '.join(chunk_summaries)
                # Summarize the combined summaries if too long
                if len(combined_summary.split()) > max_length:
                    try:
                        result = self.summarizer(
                            combined_summary,
                            max_length=max_length,
                            min_length=min_length,
                            do_sample=False
                        )
                        summary = result[0]['summary_text']
                    except Exception as e:
                        summary = combined_summary[:max_length*5]  # Rough truncation
                else:
                    summary = combined_summary
            else:
                summary = "Error: Could not process any chunks"
        
        return {
            "summary": summary,
            "chunks_processed": len(chunks),
            "original_length": original_length,
            "summary_length": len(summary.split()),
            "compression_ratio": round(original_length / len(summary.split()), 2) if summary else 0
        }
    
    def read_file(self, file_path: str) -> str:
        """Read text from various file formats."""
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if path.suffix.lower() == '.txt':
            with open(path, 'r', encoding='utf-8') as f:
                return f.read()
        
        elif path.suffix.lower() == '.pdf':
            if not PDF_AVAILABLE:
                raise ImportError("PyPDF2 not installed. Run: pip install PyPDF2")
            
            text = ""
            with open(path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    text += page.extract_text() + "\n"
            return text
        
        elif path.suffix.lower() in ['.docx', '.doc']:
            if not DOCX_AVAILABLE:
                raise ImportError("python-docx not installed. Run: pip install python-docx")
            
            doc = Document(path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        
        else:
            # Try to read as plain text
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    return f.read()
            except UnicodeDecodeError:
                with open(path, 'r', encoding='latin-1') as f:
                    return f.read()
    
    def summarize_file(self, file_path: str, **kwargs) -> Dict[str, Any]:
        """Summarize text from a file."""
        text = self.read_file(file_path)
        result = self.summarize_text(text, **kwargs)
        result["source_file"] = file_path
        return result
    
    def batch_summarize(self, texts: List[str], **kwargs) -> List[Dict[str, Any]]:
        """Summarize multiple texts in batch."""
        results = []
        for i, text in enumerate(texts):
            self.logger.info(f"Processing text {i+1}/{len(texts)}")
            result = self.summarize_text(text, **kwargs)
            result["text_index"] = i
            results.append(result)
        return results


def main():
    """Command-line interface for the text summarizer."""
    parser = argparse.ArgumentParser(description="Summarize text using Hugging Face models")
    parser.add_argument("input", help="Input text file or directory")
    parser.add_argument("-o", "--output", help="Output file (optional)")
    parser.add_argument("-m", "--model", default="facebook/bart-large-cnn", 
                       help="Hugging Face model name")
    parser.add_argument("--max-length", type=int, default=150, 
                       help="Maximum summary length")
    parser.add_argument("--min-length", type=int, default=50, 
                       help="Minimum summary length")
    parser.add_argument("--chunk-size", type=int, default=1000, 
                       help="Chunk size for long texts")
    parser.add_argument("--device", choices=["cpu", "cuda"], 
                       help="Device to use (auto-detected if not specified)")
    
    args = parser.parse_args()
    
    # Initialize summarizer
    try:
        summarizer = TextSummarizer(model_name=args.model, device=args.device)
    except Exception as e:
        print(f"Failed to initialize summarizer: {e}")
        return 1
    
    # Process input
    try:
        if os.path.isfile(args.input):
            # Single file
            result = summarizer.summarize_file(
                args.input,
                max_length=args.max_length,
                min_length=args.min_length,
                chunk_size=args.chunk_size
            )
            
            # Display results
            print(f"\n{'='*60}")
            print(f"SUMMARY OF: {args.input}")
            print(f"{'='*60}")
            print(f"Original length: {result['original_length']} words")
            print(f"Summary length: {result['summary_length']} words")
            print(f"Compression ratio: {result['compression_ratio']}:1")
            print(f"Chunks processed: {result['chunks_processed']}")
            print(f"\n{'-'*60}")
            print("SUMMARY:")
            print(f"{'-'*60}")
            print(result['summary'])
            print(f"{'='*60}\n")
            
            # Save output if specified
            if args.output:
                with open(args.output, 'w', encoding='utf-8') as f:
                    f.write(f"Summary of: {args.input}\n")
                    f.write(f"Generated by: {args.model}\n")
                    f.write(f"{'='*60}\n")
                    f.write(result['summary'])
                print(f"Summary saved to: {args.output}")
        
        elif os.path.isdir(args.input):
            # Directory processing
            text_files = []
            for ext in ['*.txt', '*.pdf', '*.docx']:
                text_files.extend(Path(args.input).glob(ext))
            
            if not text_files:
                print(f"No supported text files found in: {args.input}")
                return 1
            
            print(f"Found {len(text_files)} files to process...")
            
            for file_path in text_files:
                try:
                    result = summarizer.summarize_file(
                        str(file_path),
                        max_length=args.max_length,
                        min_length=args.min_length,
                        chunk_size=args.chunk_size
                    )
                    
                    print(f"\n{file_path.name}:")
                    print(f"  Original: {result['original_length']} words")
                    print(f"  Summary: {result['summary_length']} words")
                    print(f"  Ratio: {result['compression_ratio']}:1")
                    print(f"  Preview: {result['summary'][:100]}...")
                    
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
        
        else:
            print(f"Input not found: {args.input}")
            return 1
            
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())


# Example usage as a library:
"""
from text_summarizer import TextSummarizer

# Initialize
summarizer = TextSummarizer()

# Summarize text
text = "Your long article text here..."
result = summarizer.summarize_text(text, max_length=200)
print(result['summary'])

# Summarize file
result = summarizer.summarize_file("article.txt")
print(f"Summary: {result['summary']}")
print(f"Compression: {result['compression_ratio']}:1")

# Batch processing
texts = ["Text 1...", "Text 2...", "Text 3..."]
results = summarizer.batch_summarize(texts)
for i, result in enumerate(results):
    print(f"Summary {i+1}: {result['summary']}")
"""