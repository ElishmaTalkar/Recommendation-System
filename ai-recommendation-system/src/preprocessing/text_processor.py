"""Custom text preprocessing pipeline for production-quality NLP."""
import re
import logging
from typing import List, Optional
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextPreprocessor:
    """
    Production-quality text preprocessor with configurable pipeline.
    
    Features:
    - HTML/URL removal
    - Lowercasing and punctuation handling
    - Stopword removal with custom additions
    - Lemmatization using NLTK (simplified for compatibility)
    - Configurable pipeline stages
    """
    
    def __init__(
        self,
        lowercase: bool = True,
        remove_urls: bool = True,
        remove_html: bool = True,
        remove_stopwords: bool = True,
        lemmatize: bool = True,
        custom_stopwords: Optional[List[str]] = None
    ):
        """
        Initialize the text preprocessor.
        
        Args:
            lowercase: Convert text to lowercase
            remove_urls: Remove URLs from text
            remove_html: Remove HTML tags
            remove_stopwords: Remove stopwords
            lemmatize: Apply lemmatization
            custom_stopwords: Additional stopwords to remove
        """
        self.lowercase = lowercase
        self.remove_urls = remove_urls
        self.remove_html = remove_html
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
        
        # Initialize stopwords
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            logger.info("Downloading NLTK stopwords...")
            nltk.download('stopwords', quiet=True)
        
        self.stopwords = set(stopwords.words('english'))
        if custom_stopwords:
            self.stopwords.update(custom_stopwords)
        
        # Initialize lemmatizer
        if self.lemmatize:
            try:
                nltk.data.find('corpora/wordnet')
            except LookupError:
                logger.info("Downloading NLTK wordnet...")
                nltk.download('wordnet', quiet=True)
                nltk.download('omw-1.4', quiet=True)
            
            self.lemmatizer = WordNetLemmatizer()
        else:
            self.lemmatizer = None
    
    def clean_html(self, text: str) -> str:
        """Remove HTML tags from text."""
        html_pattern = re.compile(r'<.*?>')
        return html_pattern.sub('', text)
    
    def clean_urls(self, text: str) -> str:
        """Remove URLs from text."""
        url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        return url_pattern.sub('', text)
    
    def clean_punctuation(self, text: str) -> str:
        """Remove excessive punctuation while preserving word boundaries."""
        # Remove special characters but keep spaces
        text = re.sub(r'[^\w\s]', ' ', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def remove_stopwords_from_text(self, text: str) -> str:
        """Remove stopwords from text."""
        words = text.split()
        filtered_words = [word for word in words if word.lower() not in self.stopwords]
        return ' '.join(filtered_words)
    
    def lemmatize_text(self, text: str) -> str:
        """Apply lemmatization using NLTK."""
        if not self.lemmatizer:
            return text
        
        try:
            words = text.split()
            lemmatized = [self.lemmatizer.lemmatize(word) for word in words]
            return ' '.join(lemmatized)
        except Exception as e:
            logger.error(f"Lemmatization error: {e}")
            return text
    
    def preprocess(self, text: str) -> str:
        """
        Apply full preprocessing pipeline to text.
        
        Args:
            text: Input text to preprocess
            
        Returns:
            Preprocessed text
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Apply preprocessing stages
        if self.remove_html:
            text = self.clean_html(text)
        
        if self.remove_urls:
            text = self.clean_urls(text)
        
        if self.lowercase:
            text = text.lower()
        
        text = self.clean_punctuation(text)
        
        if self.remove_stopwords:
            text = self.remove_stopwords_from_text(text)
        
        if self.lemmatize:
            text = self.lemmatize_text(text)
        
        return text.strip()
    
    def preprocess_batch(self, texts: List[str]) -> List[str]:
        """
        Preprocess a batch of texts.
        
        Args:
            texts: List of texts to preprocess
            
        Returns:
            List of preprocessed texts
        """
        return [self.preprocess(text) for text in texts]
    
    def __call__(self, text: str) -> str:
        """Allow the preprocessor to be called as a function."""
        return self.preprocess(text)


# Example usage
if __name__ == "__main__":
    # Initialize preprocessor
    preprocessor = TextPreprocessor()
    
    # Test text
    test_text = """
    <p>Check out this amazing product at https://example.com!</p>
    The RUNNING shoes are designed for MAXIMUM comfort and performance.
    """
    
    # Preprocess
    cleaned = preprocessor.preprocess(test_text)
    print(f"Original: {test_text}")
    print(f"Cleaned: {cleaned}")



# Example usage
if __name__ == "__main__":
    # Initialize preprocessor
    preprocessor = TextPreprocessor()
    
    # Test text
    test_text = """
    <p>Check out this amazing product at https://example.com!</p>
    The RUNNING shoes are designed for MAXIMUM comfort and performance.
    """
    
    # Preprocess
    cleaned = preprocessor.preprocess(test_text)
    print(f"Original: {test_text}")
    print(f"Cleaned: {cleaned}")
