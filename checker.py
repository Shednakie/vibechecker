from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import numpy as np

def main():
    text = "Today sucks ass"
    sentiment_analysis(text)

# sentiment analysis
def sentiment_analysis(text):
    # preprocess text
    text_content = []

    for word in text.split(' '):
        if word.startswith('@') and len(word) > 1:
            word = '@user'
        elif word.startswith('http'):
            word = "*http link*"
        text_content.append(word)

    processed_content = " ".join(text_content)

    # load model
    roberta = "cardiffnlp/twitter-roberta-base-sentiment"
    model = AutoModelForSequenceClassification.from_pretrained(roberta)
    tokenizer = AutoTokenizer.from_pretrained(roberta)

    encoded_content = tokenizer(processed_content, return_tensors='pt')
    output = model(encoded_content['input_ids'], encoded_content['attention_mask'])
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    max_score = np.max(scores)
    max_score = round(max_score * 100, 2)
    score_index = np.argmax(scores)
    # print(f"{max_score}% {labels[score_index]}")
    return max_score, score_index

if __name__ == "__main__":
    main()