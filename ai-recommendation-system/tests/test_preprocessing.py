"""Tests for text preprocessing."""
import pytest
from src.preprocessing.text_processor import TextPreprocessor


def test_html_removal():
    """Test HTML tag removal."""
    preprocessor = TextPreprocessor(remove_html=True, lemmatize=False)
    
    text = "<p>This is a <strong>test</strong> paragraph.</p>"
    result = preprocessor.preprocess(text)
    
    assert "<p>" not in result
    assert "<strong>" not in result
    assert "test" in result


def test_url_removal():
    """Test URL removal."""
    preprocessor = TextPreprocessor(remove_urls=True, lemmatize=False)
    
    text = "Check out https://example.com for more info"
    result = preprocessor.preprocess(text)
    
    assert "https://example.com" not in result
    assert "check" in result


def test_lowercase():
    """Test lowercasing."""
    preprocessor = TextPreprocessor(lowercase=True, lemmatize=False)
    
    text = "HELLO World"
    result = preprocessor.preprocess(text)
    
    assert result == "hello world"


def test_stopword_removal():
    """Test stopword removal."""
    preprocessor = TextPreprocessor(
        remove_stopwords=True,
        lemmatize=False,
        lowercase=True
    )
    
    text = "this is a test of the system"
    result = preprocessor.preprocess(text)
    
    # Common stopwords should be removed
    assert "this" not in result
    assert "is" not in result
    assert "a" not in result
    assert "the" not in result
    
    # Content words should remain
    assert "test" in result
    assert "system" in result


def test_empty_input():
    """Test handling of empty input."""
    preprocessor = TextPreprocessor()
    
    assert preprocessor.preprocess("") == ""
    assert preprocessor.preprocess(None) == ""


def test_batch_processing():
    """Test batch preprocessing."""
    preprocessor = TextPreprocessor(lemmatize=False)
    
    texts = [
        "First text",
        "Second text",
        "Third text"
    ]
    
    results = preprocessor.preprocess_batch(texts)
    
    assert len(results) == 3
    assert all(isinstance(r, str) for r in results)


def test_custom_stopwords():
    """Test custom stopwords."""
    custom_stopwords = ["custom", "word"]
    preprocessor = TextPreprocessor(
        remove_stopwords=True,
        custom_stopwords=custom_stopwords,
        lemmatize=False,
        lowercase=True
    )
    
    text = "this is a custom word test"
    result = preprocessor.preprocess(text)
    
    assert "custom" not in result
    assert "word" not in result
    assert "test" in result
