from flask import Flask, request, jsonify
import numpy as np
import joblib

app = Flask(__name__)

# TODO: Load the trained model from the shared volume (use the correct path)
model = joblib.load('/app/models/iris_model.pkl')

def _bad_request(msg: str, code: int = 400):
    return jsonify({"error": msg}), code

def _to_input(payload):
    # Validate JSON body
    if payload is None:
        raise ValueError("Invalid or missing JSON body. Send application/json with an 'input' field.")
    if "input" not in payload:
        raise KeyError("Request JSON must include key 'input' as a list of numbers.")
    x = payload["input"]
    if not isinstance(x, (list, tuple)):
        raise TypeError("'input' must be a list (or tuple) of numbers.")
    if len(x) == 0:
        raise ValueError("'input' cannot be empty.")
    try:
        arr = np.array(x, dtype=float).reshape(1, -1)
    except Exception:
        raise ValueError("'input' must contain only numeric values and be convertible to floats.")
    return arr

# TODO: Add request method to predict
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # TODO: Get the input array from the request body and make prediction using the model
        get_json = request.get_json(force=True, silent=True)
        iris_input = _to_input(get_json)

        # HINT: use np.array().reshape(1, -1) to convert input to 2D array
        prediction = model.predict(iris_input)[0]

        return jsonify({"pred": str(prediction)}), 200
    
    except KeyError as e:
        return _bad_request(str(e), 400)
    except (TypeError, ValueError) as e:
        return _bad_request(str(e), 400)
    except Exception as e:
        # Last-resort catch: avoid leaking stack traces to clients
        return _bad_request(f"Unhandled error: {str(e)}", 500)

@app.route('/')
def hello():
    return 'Welcome to Docker Lab'

@app.errorhandler(404)
def not_found(_e):
    return jsonify({"error": "Not Found"}), 404

@app.errorhandler(405)
def method_not_allowed(_e):
    return jsonify({"error": "Method Not Allowed"}), 405

@app.errorhandler(500)
def internal_error(_e):
    return jsonify({"error": "Internal Server Error"}), 500

if __name__ == '__main__':
    #Run the Flask app (bind it to port 8080 or any other port)
    app.run(debug=True, port=8080, host='0.0.0.0')
