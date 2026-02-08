"""
Machine Learning Fake News Detection Bot
Uses TF-IDF + Logistic Regression trained on embedded sample data.
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import numpy as np

# Sample training data (embedded in code for simplicity)
# In real usage, you'd train offline and save model.pkl
TRAINING_DATA = [
    # Fake news examples
    ("SHOCKING: Scientists discover miracle cure hidden by government!", 1),
    ("You won't BELIEVE what this celebrity did - doctors HATE this trick!", 1),
    ("SECRET conspiracy revealed: The truth they don't want you to know", 1),
    ("BREAKING: Miracle weight loss pill works overnight - no exercise needed!", 1),
    ("Exposed: The shocking truth about vaccines that media won't tell you", 1),
    ("Unbelievable discovery proves everything we knew was WRONG!", 1),
    ("This weird trick will make you rich overnight - bankers hate it!", 1),
    ("Government hiding alien technology in secret underground base", 1),
    ("URGENT: Share this before it gets deleted by the deep state!", 1),
    ("Exposed: Famous politician secretly a lizard person!", 1),
    
    # Real news examples  
    ("Stock market closes higher amid positive economic indicators", 0),
    ("City council approves new budget for public transportation", 0),
    ("Scientists publish study on climate change effects in peer-reviewed journal", 0),
    ("Local hospital announces expansion of emergency services", 0),
    ("Technology company reports quarterly earnings above expectations", 0),
    ("Weather service predicts moderate temperatures for the week", 0),
    ("University researchers develop new method for water purification", 0),
    ("Government announces new policy on renewable energy incentives", 0),
    ("Central bank maintains interest rates amid stable inflation", 0),
    ("Sports team wins championship after close finale match", 0),
]

# Train model at import time
_vectorizer = TfidfVectorizer(max_features=500, stop_words='english', ngram_range=(1, 2))
_texts = [t[0] for t in TRAINING_DATA]
_labels = [t[1] for t in TRAINING_DATA]
_X = _vectorizer.fit_transform(_texts)
_model = LogisticRegression(random_state=42)
_model.fit(_X, _labels)

print("ML Bot: Model trained successfully!")


def analyze(article_title: str, article_content: str) -> dict:
    """
    Analyze a news article using TF-IDF + Logistic Regression.
    
    Args:
        article_title: The headline of the article
        article_content: The body text of the article
        
    Returns:
        dict with 'is_fake' (bool) and 'confidence' (float 0-1)
    """
    # Combine title and content
    full_text = f"{article_title} {article_content}"
    
    # Transform using trained vectorizer
    features = _vectorizer.transform([full_text])
    
    # Predict
    prediction = _model.predict(features)[0]
    probabilities = _model.predict_proba(features)[0]
    
    # Get confidence (probability of the predicted class)
    confidence = float(max(probabilities))
    
    return {
        "is_fake": bool(prediction == 1),
        "confidence": confidence
    }


# Local testing
if __name__ == "__main__":
    # Test with fake news
    result1 = analyze(
        "SHOCKING: Secret cure discovered!",
        "Scientists have found a miracle cure that the government is hiding from you."
    )
    print(f"Fake news test: {result1}")
    
    # Test with real news
    result2 = analyze(
        "City approves new budget",
        "The city council voted to approve the annual budget for infrastructure projects."
    )
    print(f"Real news test: {result2}")
