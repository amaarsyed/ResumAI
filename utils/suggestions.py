import re
import spacy

# Load spaCy English model (run python -m spacy download en_core_web_sm if needed)
nlp = spacy.load("en_core_web_sm")

def extract_keywords(text):
    """
    Extract meaningful nouns and skills from text.
    Args:
        text (str): Input resume or job description text.
    Returns:
        set: Set of keywords/skills
    """
    doc = nlp(text.lower())
    keywords = set()

    for token in doc:
        if token.pos_ in ["NOUN", "PROPN", "ADJ"] and not token.is_stop and len(token.text) > 2:
            keywords.add(token.lemma_)

    return keywords

def get_missing_keywords(resume_text, jd_text):
    """
    Compare JD and resume keywords and return what's missing from the resume.
    """
    jd_keywords = extract_keywords(jd_text)
    resume_keywords = extract_keywords(resume_text)
    missing = jd_keywords - resume_keywords

    return list(missing)
