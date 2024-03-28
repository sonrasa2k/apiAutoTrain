from flask import Flask, request, jsonify

HOST ="http://localhost:9999"

app = Flask(__name__)

@app.route("/api/ai/callback",methods=['POST'])
def get_callback():
    request_data = request.get_json()
    print(request_data)
    return request_data

@app.route("/api/ai/callback",methods=['POST'])
def get_callback2():
    request_data = request.get_json()
    print(request_data)
    return request_data
if __name__ == '__main__':
    app.run(port=9999)