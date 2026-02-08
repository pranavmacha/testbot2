# ML Fake News Detection Bot

A simple machine learning bot that uses TF-IDF vectorization and Logistic Regression to detect fake news.

## How it works

1. **Training Data**: The bot is trained on a small embedded dataset of 20 examples (10 fake, 10 real)
2. **TF-IDF**: Converts text into numerical features using word frequency analysis
3. **Logistic Regression**: Classifies articles as fake or real

## Files

- `bot.py` - Main bot code with `analyze()` function
- `requirements.txt` - Dependencies (scikit-learn, numpy)

## Usage

Push this folder to GitHub and paste the URL in Bot Arena!

## Improving the model

For better accuracy, you could:
- Add more training data
- Use a pre-trained model file (model.pkl)
- Try different classifiers (RandomForest, SVM, etc.)
