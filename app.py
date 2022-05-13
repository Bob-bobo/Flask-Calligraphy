from flask import Flask,make_response,render_template,request
from gevent import pywsgi
import calligraphy3
app = Flask(__name__)


@app.route('/')
def hello_world():
    res = make_response(render_template("index2.html"))
    return res

strFilepath = ''

@app.route('/uploadImg', methods=['GET','POST'])
def upload_img():
    if request.method == 'GET':
        return make_response(render_template("index2.html"))
    elif request.method == 'POST':
        if 'myImg' in request.files:
            objFile = request.files.get('myImg')
            print("进来网址了")
            strFilename = objFile.filename
            strFilepath = "./static/images/" + strFilename
            objFile.save(strFilepath)
            calligraphy3.perdict_img(strFilepath)
            return 'file saved'
        else:
            return 'err'
    else:
        return 'err'

if __name__ == '__main__':
    from waitress import serve
    serve(app, host="0.0.0.0", port=80)
    # app.run(host="0.0.0.0", port=80)
    # server = pywsgi.WSGIServer(('0.0.0.0', 80), app)
    # server.serve_forever()
