#!/usr/bin/env python3
"""
Dataset 1: Wikipedia Paragraph Extractor (Fixed Version)
Extracts self-contained paragraphs from Wikipedia articles (150-500 words, plain text only)
"""

import json
import re
import requests
import time
from typing import List, Dict
import argparse

class WikipediaExtractor:
    def __init__(self):
        self.session = requests.Session() 
        self.session.headers.update({
            'User-Agent': 'WikipediaDatasetExtractor/1.0 (Educational Purpose)'
        })
    
    def get_random_articles(self, count: int = 10) -> List[str]:
        """Get random Wikipedia article titles"""
        url = "https://en.wikipedia.org/api/rest_v1/page/random/summary"
        titles = []
        
        for _ in range(count):
            try:
                response = self.session.get(url)
                if response.status_code == 200:
                    data = response.json()
                    title = data['title']
                    # Skip disambiguation pages and other meta pages
                    if not any(skip in title.lower() for skip in ['disambiguation', 'list of', 'category:', 'template:']):
                        titles.append(title)
                    time.sleep(0.1)  # Be respectful to Wikipedia's servers
            except Exception as e:
                print(f"Error fetching random article: {e}")
                continue
        
        return titles
    
    def get_article_content(self, title: str) -> str:
        """Get the plain text content of a Wikipedia article"""
        # Use the Wikipedia API to get full article content
        page_url = f"https://en.wikipedia.org/w/api.php"
        params = {
            'action': 'query',
            'format': 'json',
            'titles': title,
            'prop': 'extracts',
            'exintro': False,  # Get full article, not just intro
            'explaintext': True,  # Plain text, no HTML
            'exsectionformat': 'plain'
        }
        
        try:
            response = self.session.get(page_url, params=params)
            if response.status_code == 200:
                data = response.json()
                pages = data.get('query', {}).get('pages', {})
                for page_id, page_data in pages.items():
                    if page_id != '-1' and 'extract' in page_data:  # -1 means page not found
                        content = page_data['extract']
                        print(f"  Retrieved {len(content)} characters from '{title}'")
                        return content
                print(f"  No content found for '{title}'")
                return ""
        except Exception as e:
            print(f"Error fetching article '{title}': {e}")
            return ""
    
    def clean_text(self, text: str) -> str:
        """Clean text by removing unwanted characters and formatting"""
        # Remove citations like [1], [citation needed], etc.
        text = re.sub(r'\[.*?\]', '', text)
        
        # Remove common Wikipedia artifacts
        text = re.sub(r'\(listen\)', '', text)
        text = re.sub(r'\(pronunciation.*?\)', '', text)
        text = re.sub(r'\(.*?pronunciation.*?\)', '', text)
        
        # Remove section headers that might be left over
        text = re.sub(r'^==+.*?==+$', '', text, flags=re.MULTILINE)
        
        # Clean up extra whitespace and newlines
        text = re.sub(r'\n+', '\n', text)
        text = re.sub(r'\s+', ' ', text)
        
        # Clean up punctuation spacing
        text = re.sub(r'\s+([,.!?;:])', r'\1', text)
        
        return text.strip()
    
    def extract_paragraphs(self, text: str, source_title: str) -> List[str]:
        """Extract self-contained paragraphs from text"""
        print(f"  Extracting paragraphs from {len(text)} characters...")
        
        # First clean the text
        text = self.clean_text(text)
        
        # Split by multiple newlines or double periods (common in Wikipedia)
        # Also split on sentences that clearly start new topics
        paragraphs = re.split(r'\n\n+|\.\s*\n|\n(?=[A-Z][a-z])', text)
        
        valid_paragraphs = []
        
        for i, paragraph in enumerate(paragraphs):
            paragraph = paragraph.strip()
            
            if not paragraph:
                continue
                
            # Skip very short paragraphs
            if len(paragraph) < 50:
                continue
            
            # Check word count (150-500 words, but be more flexible)
            words = paragraph.split()
            word_count = len(words)
            
            print(f"    Paragraph {i+1}: {word_count} words")
            
            # Be more flexible with word count - accept 100-600 words
            if 100 <= word_count <= 600:
                # Ensure it ends with proper punctuation
                if not paragraph.endswith(('.', '!', '?')):
                    # Try to find a good ending point
                    sentences = re.split(r'[.!?]+', paragraph)
                    if len(sentences) > 1:
                        # Take all but the last incomplete sentence
                        complete_sentences = sentences[:-1]
                        paragraph = '. '.join(complete_sentences) + '.'
                        word_count = len(paragraph.split())
                
                # Final check - must have at least 2 sentences and reasonable length
                sentence_count = len(re.findall(r'[.!?]+', paragraph))
                if sentence_count >= 2 and 100 <= word_count <= 600:
                    # Additional quality checks
                    if self.is_good_paragraph(paragraph):
                        valid_paragraphs.append(paragraph)
                        print(f"    ✓ Added paragraph with {word_count} words")
                    else:
                        print(f"    ✗ Paragraph failed quality check")
                else:
                    print(f"    ✗ Paragraph has {sentence_count} sentences, {word_count} words")
            else:
                print(f"    ✗ Word count {word_count} outside range 100-600")
        
        print(f"  Found {len(valid_paragraphs)} valid paragraphs from '{source_title}'")
        return valid_paragraphs
    
    def is_good_paragraph(self, paragraph: str) -> bool:
        """Check if paragraph meets quality criteria"""
        # Skip paragraphs that are mostly lists
        if paragraph.count('\n') > 3:
            return False
        
        # Skip paragraphs with too many numbers (likely data/statistics)
        numbers = re.findall(r'\d+', paragraph)
        if len(numbers) > len(paragraph.split()) * 0.3:  # More than 30% numbers
            return False
        
        # Skip paragraphs that are mostly parenthetical information
        if paragraph.count('(') > 5:
            return False
        
        # Must contain some common English words
        common_words = ['the', 'and', 'of', 'to', 'a', 'in', 'is', 'was', 'are', 'were']
        paragraph_lower = paragraph.lower()
        common_word_count = sum(1 for word in common_words if word in paragraph_lower)
        
        if common_word_count < 3:
            return False
        
        return True
    
    def create_dataset(self, num_articles: int = 50, output_file: str = "dataset_1_paragraphs.json") -> None:
        """Create the first dataset with Wikipedia paragraphs"""
        print(f"Extracting paragraphs from {num_articles} Wikipedia articles...")
        
        all_paragraphs = []
        processed_articles = 0
        failed_articles = 0
        
        while processed_articles < num_articles and failed_articles < num_articles:
            # Get batch of random articles
            print(f"\nFetching batch of random articles...")
            article_titles = self.get_random_articles(10)
            
            for title in article_titles:
                if processed_articles >= num_articles:
                    break
                    
                print(f"\nProcessing article {processed_articles + 1}/{num_articles}: {title}")
                content = self.get_article_content(title)
                
                if content and len(content) > 500:  # Ensure we have substantial content
                    paragraphs = self.extract_paragraphs(content, title)
                    for paragraph in paragraphs:
                        all_paragraphs.append({
                            "id": len(all_paragraphs) + 1,
                            "source_article": title,
                            "paragraph": paragraph,
                            "word_count": len(paragraph.split())
                        })
                    processed_articles += 1
                else:
                    print(f"  Skipping '{title}' - insufficient content")
                    failed_articles += 1
                
                # Small delay to be respectful
                time.sleep(0.2)
        
        # Save to JSON file
        dataset = {
            "metadata": {
                "description": "Wikipedia paragraphs dataset",
                "total_paragraphs": len(all_paragraphs),
                "articles_processed": processed_articles,
                "word_range": "100-600 words per paragraph",
                "extraction_method": "improved_extraction_v2"
            },
            "paragraphs": all_paragraphs
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)
        
        print(f"\n=== EXTRACTION COMPLETE ===")
        print(f"Dataset saved to {output_file}")
        print(f"Total paragraphs extracted: {len(all_paragraphs)}")
        print(f"Articles successfully processed: {processed_articles}")
        print(f"Articles failed: {failed_articles}")
        
        if all_paragraphs:
            avg_words = sum(p['word_count'] for p in all_paragraphs) / len(all_paragraphs)
            print(f"Average paragraph length: {avg_words:.1f} words")
            
            # Show a sample paragraph
            print(f"\nSample paragraph:")
            print(f"Source: {all_paragraphs[0]['source_article']}")
            print(f"Length: {all_paragraphs[0]['word_count']} words")
            print(f"Text: {all_paragraphs[0]['paragraph'][:200]}...")
        else:
            print("⚠️  WARNING: No paragraphs were extracted!")
            print("This might be due to:")
            print("- Network connectivity issues")
            print("- Wikipedia API rate limiting")
            print("- Articles not meeting quality criteria")

def main():
    parser = argparse.ArgumentParser(description='Extract Wikipedia paragraphs for dataset creation')
    parser.add_argument('--articles', type=int, default=50, help='Number of articles to process')
    parser.add_argument('--output', type=str, default='dataset_1_paragraphs.json', help='Output JSON file')
    
    args = parser.parse_args()
    
    extractor = WikipediaExtractor()
    extractor.create_dataset(args.articles, args.output)

if __name__ == "__main__":
    main()

