from flask import Flask
from flask_cors import CORS
from routes import main  # Import the blueprint from routes.py

app = Flask(__name__)
CORS(app)

# Register the routes blueprint
app.register_blueprint(main)

if __name__ == '__main__':
    app.run(debug=True, port=8001)