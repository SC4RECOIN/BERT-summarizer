from flask import Flask, jsonify, request
from waitress import serve
from summarizer import AbstractSummarizer
import traceback


def create_app(model: AbstractSummarizer):
    app = Flask(__name__)

    @app.route('/', methods=['GET'])
    def ping():
        print("Pong.")
        return jsonify({'message': 'Summarization app is running'}), 200

    @app.route('/', methods=['POST'])
    def summarize():
        print("Summarization endpoint hit")
        try:
            texts = request.json['text']
            if type(texts) == str:
                texts = [texts]

            summaries = summarizer.get_summary(texts)

            return jsonify({
                'message': 'Summarization successful',
                'summary': summaries
            }), 200
        except Exception as e:
            tb = traceback.format_exc()
            print(f"TRACEBACK:\n\n{tb}\n")
            return jsonify({'message': str(e), 'stacktrace': str(tb)}), 500

    return app


if __name__ == "__main__":
    summarizer = AbstractSummarizer()
    print('Summarization model ready')
    serve(create_app(summarizer), host='0.0.0.0', port=5000)
