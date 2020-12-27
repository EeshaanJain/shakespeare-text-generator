from flask import Flask, render_template, request
import tensorflow as tf
from predict import generate_text

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

model = tf.keras.models.load_model('model1.h5')


app = Flask(__name__, template_folder='templates')


@app.route('/', methods=['GET', 'POST'])
def main():
    if request.method == 'GET':
        return (render_template('main.html'))

    if request.method == 'POST':
        start_string = request.form['start_string']
        prediction = generate_text(model, start_string)
        prediction = prediction.replace('\n', '<br>')
        return render_template('main.html', original_input=start_string,
                               result=prediction)


if __name__ == '__main__':
    app.run()
