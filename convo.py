import random
import numpy as np
import json
import torch
import random
from model import LSTM
from pre_processing import bag_of_words, tokenization, normalization
from flask import Flask, Response,render_template, jsonify, request
from flask_cors import CORS, cross_origin


try:
    device = torch.device("cpu")

    with open('data1.json', 'r') as instances:
        data = json.load(instances)

    FILE = "dataserialized1.pth"
    dataserialized = torch.load(FILE)

    seq_length = dataserialized["seq_length"]
    input_size = dataserialized["input_size"]
    hidden_size = dataserialized["hidden_size"]
    num_layers = dataserialized["num_layers"]
    num_classes = dataserialized["num_classes"]
    word_list = dataserialized["word_list"]
    tags = dataserialized["tags"]
    model_state = dataserialized["model_state"]

    model = LSTM(seq_length, input_size, hidden_size, num_layers, num_classes).to(device)
    model.load_state_dict(model_state)
    model.eval()
except Exception as e:
    print(e)
    history = []

app = Flask(__name__)
CORS(app, resourses={r"/api/*":{"origins": "127.0.0.1:3000"}})
app.config['CORS_HEADERS']= 'Content-Type'

@app.route('/api')
@cross_origin()
def hello_world():
    return "hello1"

# @app.route('/api/test', methods=['GET'])
# @cross_origin()
# def hello_world():
#     return "hello2"
#convo one
@app.route("/api/message1", methods=['POST'])
@cross_origin()
def get_bot_response():
 if request.method == "POST":
    bot = "Convo"
    user_data = request.json

    sentence = user_data['message']#  while True:# response=input("You:")
    sentence = normalization(sentence)
    sentence = tokenization(sentence)
    # print(sentence)
    # print(word_list)
    # return jsonify(convo_response="Bot started 2...")

    x = bag_of_words(sentence, word_list)
    x = torch.from_numpy(x)
    x = x.reshape(-1, x.shape[0])
    x = x.to(device)# x=torch.tensor(x)# print(x.shape)

    output, hidden = model(x)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    prob = torch.softmax(output, dim=1)
    probability = prob[0][predicted.item()]

    if (probability.item() > 0.80):

        for i in data['data']:
            if tag == i['tag']:
                # print(f"{bot}: {random.choice(i['bot_responses'])}")
                #  response_payload = f"User: {userreply} \n{bot}: {random.choice(i['bot_responses'])}\n"
                #  history.append(response_payload)
                #  return Response(history,  mimetype='text/plain')
                return jsonify(random.choice(i['bot_responses']))
    else:
        return jsonify("I do not understand...")


# convo two
@app.route("/api/message2", methods=['POST'])
@cross_origin()
def get_bot_response2():
    try:
        device = torch.device("cpu")

        with open('data2.json', 'r') as instances:
            data = json.load(instances)

        FILE = "dataserialized2.pth"
        dataserialized = torch.load(FILE)

        seq_length = dataserialized["seq_length"]
        input_size = dataserialized["input_size"]
        hidden_size = dataserialized["hidden_size"]
        num_layers = dataserialized["num_layers"]
        num_classes = dataserialized["num_classes"]
        word_list = dataserialized["word_list"]
        tags = dataserialized["tags"]
        model_state = dataserialized["model_state"]

        model = LSTM(seq_length, input_size, hidden_size, num_layers, num_classes).to(device)
        model.load_state_dict(model_state)
        model.eval()
    except Exception as e:
        print(e)
    if request.method == "POST":
        bot = "Convo"
        user_data = request.json

        sentence = user_data['message']  # 
        sentence = normalization(sentence)
        sentence = tokenization(sentence)
        x = bag_of_words(sentence, word_list)
        x = torch.from_numpy(x)
        x = x.reshape(-1, x.shape[0])
        x = x.to(device)  # x=torch.tensor(x)# print(x.shape)

        output, hidden = model(x)
        _, predicted = torch.max(output, dim=1)
        tag = tags[predicted.item()]

        prob = torch.softmax(output, dim=1)
        probability = prob[0][predicted.item()]

        if (probability.item() > 0.80):

            for i in data['data']:
                if tag == i['tag']:
                    return jsonify(random.choice(i['bot_responses']))
        else:
            return jsonify("I do not understand...")

# convo three
@app.route("/api/message3", methods=['POST'])
@cross_origin()
def get_bot_response3():
    try:
        device = torch.device("cpu")

        with open('data3.json', 'r') as instances:
            data = json.load(instances)

        FILE = "dataserialized3.pth"
        dataserialized = torch.load(FILE)

        seq_length = dataserialized["seq_length"]
        input_size = dataserialized["input_size"]
        hidden_size = dataserialized["hidden_size"]
        num_layers = dataserialized["num_layers"]
        num_classes = dataserialized["num_classes"]
        word_list = dataserialized["word_list"]
        tags = dataserialized["tags"]
        model_state = dataserialized["model_state"]

        model = LSTM(seq_length, input_size, hidden_size, num_layers, num_classes).to(device)
        model.load_state_dict(model_state)
        model.eval()
    except Exception as e:
        print(e)
    if request.method == "POST":
        bot = "Convo"
        user_data = request.json

        sentence = user_data['message']  
        sentence = normalization(sentence)
        sentence = tokenization(sentence)
        x = bag_of_words(sentence, word_list)
        x = torch.from_numpy(x)
        x = x.reshape(-1, x.shape[0])
        x = x.to(device)  

        output, hidden = model(x)
        _, predicted = torch.max(output, dim=1)
        tag = tags[predicted.item()]

        prob = torch.softmax(output, dim=1)
        probability = prob[0][predicted.item()]

        if (probability.item() > 0.80):

            for i in data['data']:
                if tag == i['tag']:
                    return jsonify(random.choice(i['bot_responses']))
        else:
            return jsonify("I do not understand...")




if __name__ == '__main__':
    app.run(debug=True)
