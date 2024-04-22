from flask import Flask, render_template, request

# Your existing code for the chatbot
import random
import json
import torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

app = Flask(__name__)

# Load intents and model
with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size)
model.load_state_dict(model_state)
model.eval()

bot_name = "Sam"

# Define route for home page
@app.route("/")
def home():
    return render_template("index.html")

# Define route for chat
@app.route("/chat", methods=["POST"])
def chat():
    if request.method == "POST":
        message = request.form["message"]
        if message == "quit":
            return "Bye!"
        else:
            sentence = tokenize(message)
            X = bag_of_words(sentence, all_words)
            X = X.reshape(1, X.shape[0])
            X = torch.from_numpy(X)

            output = model(X)
            _, predicted = torch.max(output, dim=1)

            tag = tags[predicted.item()]

            probs = torch.softmax(output, dim=1)
            prob = probs[0][predicted.item()]
            if prob.item() > 0.75:
                for intent in intents['intents']:
                    if tag == intent["tag"]:
                        response = random.choice(intent['responses'])
                        return response
            else:
                return "I do not understand..."

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
