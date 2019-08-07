from flask import Flask, render_template

app = Flask(__name__)


@app.route('/')
def hello_world():
    return render_template('index.html')


@app.route('/cam')
def cam():
    return render_template('camera.html')


if __name__ == '__main__':
    app.run()
