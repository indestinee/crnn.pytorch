from flask import Flask, render_template, request
import requests
from tester import web_tester
app = Flask(__name__, static_folder='static', static_url_path='')

@app.route("/", methods=['GET'])
def index():
    name, label = 'None', 'None'
    url = request.args.get('url')
    print(url)

    if url:
        try:
            name, label = web_tester(url)
        except:
            pass
    return render_template(**{
        'template_name_or_list': 'index.html', 
        'name': name,
        'label': label,
    })


app.run('0.0.0.0', 12345)

