from flask import Flask, request, render_template
from back_end import analyze, fetch_data, data_processing, stemming, polarity, sentiment

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/search', methods=['GET', 'POST'])
def search():
    if request.method == 'POST':
        term = request.form.get('term')
        report = analyze(term)
        data = {'Result': 'Sentiment analysis', 'Positive': report[2], 'Neutral': report[3], 'Negative': report[4]}
        return render_template('report.html', accuracy=report[0], report=report[1], data=data)


# @app.route('/chart', methods=['GET', 'POST'])
# def chart(report):
#     # if request.method == 'POST':
#     data = {'Result': 'Sentiment analysis', 'Positive': report[2], 'Neutral': report[3], 'Negative': report[4]}
#     return render_template('piechart.html', data=data)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)

