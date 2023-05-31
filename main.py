from flask import Flask, request, jsonify, render_template
from checker import sentiment_analysis

app = Flask(__name__)

@app.route("/")
def home():
    return "Home"

@app.route("/sentiment-analysis", methods=["POST"])
def analyse_text():
    data = request.get_json()
    labels = ["Negative", "Neutral", "Positive"]
    score, index = sentiment_analysis(data)
    #value = f'{score}% {labels[index]}'
    value = [score, labels[index]]
    return value


if __name__ == "__main__":
    app.run(debug=True)