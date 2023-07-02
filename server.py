from flask import Flask, request
from flask_cors import CORS
import datetime

import utils

from apscheduler.schedulers.background import BackgroundScheduler

app = Flask(__name__)
scheduler = BackgroundScheduler(daemon=True)

CORS(app)

# URL: http://localhost:5000/api?query=a&skip=b
@app.route('/api', methods=['GET'])
def search_api():
    query_param = request.args.get('query')
    skip = request.args.get('skip')
    if query_param is None:
        return 'Error: No string parameter provided', 400
    try:
        skip = int(skip)
    except:
        skip = 0
    results = utils.search_engine(query_param, skip)
    return results


# URL: http://localhost:5000/classifier
@app.route('/classifier', methods=['POST'])
def classifier():
    data = request.get_json()  
    text_param = data.get('text')
    if text_param is None:
        return 'Error: No string parameter provided', 400
    result = utils.classifier(text_param)
    return result


if __name__ == '__main__':
    # scheduler.add_job(utils.scrape_pureportal, "interval", minutes=3) 
    scheduler.add_job(utils.scrape_pureportal, "interval", weeks=1, start_date=datetime.datetime.now())
    scheduler.start()
    app.run(debug=False)