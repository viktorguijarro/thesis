#!/usr/bin/env python3
"""
Wikipedia Content Extractor for QA Dataset Creation
Extracts high-quality paragraphs from Wikipedia articles for question-answer generation.
"""

import requests
import json
import time
import re
import argparse
import logging
from typing import List, Dict, Any, Optional
from urllib.parse import quote
import random

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class WikipediaExtractor:
    def __init__(self, delay: float = 1.0, user_agent: str = "WikipediaQAExtractor/1.0"):
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': user_agent
        })
        self.base_url = "https://en.wikipedia.org/api/rest_v1"
        
    def get_random_articles(self, count: int = 50) -> List[str]:
        """Get random article titles from Wikipedia."""
        articles = []
        attempts = 0
        max_attempts = count * 3  # Allow for filtering
        
        while len(articles) < count and attempts < max_attempts:
            try:
                url = f"{self.base_url}/page/random/summary"
                response = self.session.get(url, timeout=10)
                response.raise_for_status()
                
                data = response.json()
                title = data.get('title', '')
                
                # Filter out unwanted pages
                if self._is_valid_article(title, data):
                    articles.append(title)
                    logger.info(f"Selected article {len(articles)}/{count}: {title}")
                
                attempts += 1
                time.sleep(self.delay)
                
            except Exception as e:
                logger.warning(f"Error fetching random article: {e}")
                attempts += 1
                time.sleep(self.delay * 2)
        
        return articles[:count]
    
    def _is_valid_article(self, title: str, data: Dict) -> bool:
        """Check if article is suitable for QA generation."""
        # Skip disambiguation pages
        if 'disambiguation' in title.lower():
            return False
        
        # Skip list pages
        if title.startswith('List of') or title.startswith('Category:'):
            return False
        
        # Skip very short articles
        if data.get('extract', '') and len(data['extract']) < 200:
            return False
        
        # Skip articles with certain patterns
        skip_patterns = ['(disambiguation)', 'Template:', 'File:', 'User:', 'Talk:']
        if any(pattern in title for pattern in skip_patterns):
            return False
        
        return True
    
    def extract_article_content(self, title: str) -> Optional[Dict[str, Any]]:
        """Extract full content from a Wikipedia article."""
        try:
            # Get article content
            encoded_title = quote(title.replace(' ', '_'))
            url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{encoded_title}"
            
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            summary_data = response.json()
            
            # Get full text content
            wiki_api_url = "https://en.wikipedia.org/w/api.php"
            params = {
                'action': 'query',
                'format': 'json',
                'titles': title,
                'prop': 'extracts',
                'exintro': False,
                'explaintext': True,
                'exsectionformat': 'plain'
            }
            
            response = self.session.get(wiki_api_url, params=params, timeout=15)
            response.raise_for_status()
            
            data = response.json()
            pages = data.get('query', {}).get('pages', {})
            
            if not pages:
                return None
            
            page_data = next(iter(pages.values()))
            full_text = page_data.get('extract', '')
            
            if not full_text or len(full_text) < 500:
                return None
            
            return {
                'title': title,
                'url': summary_data.get('content_urls', {}).get('desktop', {}).get('page', ''),
                'extract': full_text,
                'word_count': len(full_text.split())
            }
            
        except Exception as e:
            logger.error(f"Error extracting content for '{title}': {e}")
            return None
    
    def clean_text(self, text: str) -> str:
        """Clean Wikipedia text content."""
        # Remove citation markers
        text = re.sub(r'\[\d+\]', '', text)
        
        # Remove pronunciation guides
        text = re.sub(r'\([^)]*pronunciation[^)]*\)', '', text, flags=re.IGNORECASE)
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove section headers that might remain
        text = re.sub(r'^[A-Z][^.]*$', '', text, flags=re.MULTILINE)
        
        return text.strip()
    
    def extract_paragraphs(self, text: str) -> List[str]:
        """Extract quality paragraphs from article text."""
        # Split into potential paragraphs
        paragraphs = []
        
        # Split on double newlines first
        sections = text.split('\n\n')
        
        for section in sections:
            # Further split long sections
            if len(section) > 1000:
                sentences = section.split('. ')
                current_para = ""
                
                for sentence in sentences:
                    if len(current_para + sentence) < 600:
                        current_para += sentence + ". "
                    else:
                        if current_para.strip():
                            paragraphs.append(current_para.strip())
                        current_para = sentence + ". "
                
                if current_para.strip():
                    paragraphs.append(current_para.strip())
            else:
                if section.strip():
                    paragraphs.append(section.strip())
        
        # Filter paragraphs by quality
        quality_paragraphs = []
        for para in paragraphs:
            if self._is_quality_paragraph(para):
                quality_paragraphs.append(para)
        
        return quality_paragraphs
    
    def _is_quality_paragraph(self, paragraph: str) -> bool:
        """Assess paragraph quality for QA generation."""
        words = paragraph.split()
        word_count = len(words)
        
        # Word count filter
        if word_count < 100 or word_count > 600:
            return False
        
        # Sentence count filter
        sentences = paragraph.split('.')
        if len([s for s in sentences if s.strip()]) < 2:
            return False
        
        # Content quality checks
        # Too many numbers (likely statistics/data)
        numbers = re.findall(r'\d+', paragraph)
        if len(numbers) / word_count > 0.3:
            return False
        
        # Too many parentheses (likely technical)
        if paragraph.count('(') > 3:
            return False
        
        # Must contain common English words
        common_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        paragraph_words = set(word.lower().strip('.,!?;:') for word in words)
        if len(common_words & paragraph_words) < 3:
            return False
        
        return True
    
    def calculate_content_quality_score(self, paragraph: str) -> float:
        """Calculate Content Quality Score (CQS) for a paragraph."""
        words = paragraph.split()
        word_count = len(words)
        sentences = [s.strip() for s in paragraph.split('.') if s.strip()]
        sentence_count = len(sentences)
        
        # Word count score
        word_count_score = 1 if 100 <= word_count <= 600 else 0
        
        # Sentence count score  
        sentence_count_score = 1 if sentence_count >= 2 else 0
        
        # Content diversity score
        numbers = re.findall(r'\d+', paragraph)
        numerical_content_ratio = len(numbers) / word_count if word_count > 0 else 0
        
        list_indicators = paragraph.count('•') + paragraph.count('-') + paragraph.count('*')
        list_content_ratio = list_indicators / word_count if word_count > 0 else 0
        
        content_diversity_score = 1 - (numerical_content_ratio + list_content_ratio)
        content_diversity_score = max(0, min(1, content_diversity_score))
        
        # Calculate CQS
        normalisation_factor = 3
        cqs = (word_count_score * sentence_count_score * content_diversity_score) / normalisation_factor
        
        return cqs

def main():
    parser = argparse.ArgumentParser(description='Extract Wikipedia paragraphs for QA dataset')
    parser.add_argument('--articles', type=int, default=50, help='Number of articles to process')
    parser.add_argument('--output', type=str, default='dataset_1_paragraphs.json', help='Output file path')
    parser.add_argument('--delay', type=float, default=1.0, help='Delay between requests (seconds)')
    
    args = parser.parse_args()
    
    extractor = WikipediaExtractor(delay=args.delay)
    
    logger.info(f"Starting Wikipedia extraction for {args.articles} articles")
    
    # Get random articles
    article_titles = extractor.get_random_articles(args.articles)
    logger.info(f"Selected {len(article_titles)} articles")
    
    # Extract content and paragraphs
    all_paragraphs = []
    processed_articles = 0
    
    for title in article_titles:
        logger.info(f"Processing article: {title}")
        
        # Extract article content
        article_data = extractor.extract_article_content(title)
        
        if not article_data:
            logger.warning(f"Failed to extract content for: {title}")
            continue
        
        # Clean and extract paragraphs
        cleaned_text = extractor.clean_text(article_data['extract'])
        paragraphs = extractor.extract_paragraphs(cleaned_text)
        
        # Add paragraphs with metadata
        for i, paragraph in enumerate(paragraphs):
            cqs = extractor.calculate_content_quality_score(paragraph)
            
            paragraph_data = {
                'id': len(all_paragraphs) + 1,
                'source_article': title,
                'source_url': article_data['url'],
                'paragraph': paragraph,
                'word_count': len(paragraph.split()),
                'content_quality_score': cqs
            }
            
            all_paragraphs.append(paragraph_data)
        
        processed_articles += 1
        logger.info(f"Extracted {len(paragraphs)} paragraphs from '{title}' (CQS range: {min([extractor.calculate_content_quality_score(p) for p in paragraphs]) if paragraphs else 0:.3f}-{max([extractor.calculate_content_quality_score(p) for p in paragraphs]) if paragraphs else 0:.3f})")
        
        # Rate limiting
        time.sleep(extractor.delay)
    
    # Create output dataset
    output_data = {
        'metadata': {
            'description': 'Wikipedia paragraphs extracted for QA dataset creation',
            'total_articles_processed': processed_articles,
            'total_paragraphs': len(all_paragraphs),
            'extraction_date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'average_paragraph_length': sum(p['word_count'] for p in all_paragraphs) / len(all_paragraphs) if all_paragraphs else 0,
            'average_cqs': sum(p['content_quality_score'] for p in all_paragraphs) / len(all_paragraphs) if all_paragraphs else 0
        },
        'paragraphs': all_paragraphs
    }
    
    # Save to file
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Successfully extracted {len(all_paragraphs)} paragraphs from {processed_articles} articles")
    logger.info(f"Output saved to: {args.output}")
    logger.info(f"Average paragraph length: {output_data['metadata']['average_paragraph_length']:.1f} words")
    logger.info(f"Average Content Quality Score: {output_data['metadata']['average_cqs']:.3f}")

if __name__ == "__main__":
    main()
