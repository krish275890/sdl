from flask import Flask, render_template
import subprocess

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/run_script', methods=['POST'])
def run_script():
    subprocess.run(['python3', 'inference_classifier1.py'])
    return 'Script executed successfully!'

# @app.route('/run_scripta',methods=['POST'])
# def run_scripta():
#     subprocess.run(['python3', 'inference_classifier.py'])
#     return 'Script executed successfully!'

# @app.route('/run_scriptc',methods=['POST'])
# def run_scriptc():
#     subprocess.run(['python3', 'inference_classifier.py'])
#     return 'Script executed successfully!'


if __name__ == '__main__':
    app.run(debug=True)
