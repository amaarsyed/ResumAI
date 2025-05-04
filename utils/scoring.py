from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def calculate_similarity_score(resume_text, jd_text):
    """
    Calculate similarity score between resume and job description.
    Args:
        resume_text (str): Extracted text from resume.
        jd_text (str): Job description text.
    Returns:
        float: Similarity score as a percentage.
    """
    try:
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform([resume_text, jd_text])
        score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return round(score * 100, 2)
    except Exception as e:
        print("Error in similarity scoring:", e)
        return 0.0
