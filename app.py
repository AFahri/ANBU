import os
import sys
import tensorflow as tf

from flask import Flask, jsonify, request, render_template, flash
from wtforms import Form, TextField, TextAreaField, validators, SubmitField

# add directory
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)

print(current_dir, parent_dir)
sys.path.append(parent_dir)

# custom models
from CNN_Model.PREDICT import *
from CNN_Model.UTILS import get_root, load_pipeline


# load CNN_Model
ppl = PredictionPipeline(*load_pipeline(PREPROCESSOR_FILE,
                                            ARCHITECTURE_FILE,
                                            WEIGHTS_FILE))

global graph
graph = tf.get_default_graph()

# App config
app = Flask(__name__, static_folder='app/static')
app.config.from_object(__name__)
app.config['SECRET_KEY'] = '7d441f27d441f27567d441f2b6176a'


class ReusableForm(Form):
    comment = TextAreaField('Comment:', validators=[validators.required()])


def check_comment_toxicity(comment):
	score=ppl.predict(comment)
	return score


def set_toxicity_message(toxic_score):
	if toxic_score < 0.25:
		toxicity_message = 'Siap!: Komentarmu aman silahkan lanjutkan.'
	elif toxic_score >= 0.25 and toxic_score < 0.5:
		toxicity_message = 'Perhatian: Komentarmu sedikit beracun.'
	elif toxic_score >= 0.5 and toxic_score < 0.75:
		toxicity_message = 'Peringatan:  komentarmu agak beracun !'
	else:
		toxicity_message = 'Bahaya!: Komentarmu sangat Beracun!'

	return toxicity_message


@app.route('/', methods=['GET', 'POST'])
def hello():
	form = ReusableForm(request.form)

	print(form.errors)
	if request.method == 'POST':
		print("Form:")
		print(form)
		comment = request.form['comment']
		print(f"Comment:{comment}")

		if form.validate():

			with graph.as_default():
				toxic_score = ppl.predict([comment])
			toxic_score = toxic_score[0][0]

			toxicity_message = set_toxicity_message(toxic_score)
			print(f"Skor Toksisitas:{toxic_score}")
			print(f"Komentar ANBU:{toxicity_message}")

			flash(f"{toxicity_message} Toxicity Score: {toxic_score:0.4f}. " +\
				f"Komentarmu adalah: {comment}")

		else:
			flash('Error: Kolom harus diisi ')

	return render_template('index.html', form=form)


if __name__ == '__main__':
    app.run(debug=True, threaded=False)
