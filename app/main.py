from flask import Flask, render_template, request, current_app, url_for, redirect
import json
import uuid
from google.cloud import datastore
import datetime

app = Flask(__name__)

@app.route('/')
def hello():
    return "Hello World!"

def get_client():
    try:
        return datastore.Client()
    except:
        import os
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'C:/Users/alexa/google-cloud/usc-research-c087445cf499.json'
        return datastore.Client()


@app.route('/get_user_stats', methods=['POST'])
def get_user_stats():
    request_data = request.get_json()
    client = get_client()
    user_email = request_data.get('user_email', '')
    user = get_user(user_email, client)
    return str(user['total_tasks'])


#####
@app.route('/render', methods=['GET'])
def render():
    corpus = request.args.get('corpus', 'council-meetings')
    num_results = int(request.args.get('num_results', 3))
    task = request.args.get('task', 'ranking')

    ## fetch data
    client = get_client()
    query = client.query(kind='%s-unscored' % corpus)
    query.add_filter('finished', '=', False)
    results = list(query.fetch(limit=15))

    ## process data
    results = sorted(results, key=lambda x: x['score'])
    results = results[:int(num_results / 2) + num_results % 2] + results[-int(num_results / 2):]
    batch_id = str(uuid.uuid4())
    final_results = []
    for idx, item in enumerate(results):
        item['data_id'] = item.key.id
        item['batch']   = batch_id
        final_results.append(dict(item))

    ## 
    return render_template(
        'task-%s.html' % task,
        input=final_results,
        corpus=corpus,
        num_items=len(final_results)
    )

def get_user(email, client):
    """Get user. Create new entity if doesn't exist."""
    user_key = client.key('user', email)
    user = client.get(user_key)
    if user:
        return user
    e = datastore.Entity(key=user_key)
    e.update({'total_tasks': 0})
    client.put(e)
    return e 

#### 
@app.route('/post', methods=['POST'])
def post():
    output_data = request.get_json()
    ## 
    client = get_client()
    corpus = output_data['corpus']
    crowd_data = output_data['data']
    for item in crowd_data:
        ### add scored data
        scored_item_key = client.key('%s-scored' % corpus, item['data_id'])
        entity = datastore.Entity(
            key=scored_item_key,
            exclude_from_indexes=['text']
        )
        item['timestamp'] = str(datetime.datetime.now())
        entity.update(item)
        client.put(entity)

        ## update unscored data
        unscored_item_key = client.key('%s-unscored' % corpus, int(item['data_id']))
        unscored_item = client.get(unscored_item_key)
        unscored_item['num_completed'] += 1
        unscored_item['finished'] = unscored_item['num_completed'] >= 2
        client.put(unscored_item)

    ## update user stats
    if len(crowd_data) > 0:
        user_email = crowd_data[0]['user_email']
        user = get_user(user_email, client)
        user['total_tasks'] += 1
        client.put(user)

    ##
    return "no errors" # render_template('success-%s.html' % task)


if __name__ == '__main__':
    app.run(port=8080, debug=True)