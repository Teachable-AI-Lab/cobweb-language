from tqdm import tqdm
import pandas as pd
import argparse
from random import random
import datetime
from utility import question2instance, lemmatized_options
from cobweb.cobweb import CobwebTree

parser = argparse.ArgumentParser()
parser.add_argument('--level', type=str, default="both", choices=["leaf", "basic", "both"], help='prediction levels')
parser.add_argument('--split', type=int, default=11, help='the tested split index, start with 0')
parser.add_argument('--instances', type=int, default=1300000, help='the number of instances trained in the loaded model')
args = parser.parse_args()

# Configuration:
verbose = True
window = 10  # the size of the "gram" (so context words = 2 * window)
seed = 123  # random seed for shuffling instances
least_freq = 3
pred_level = args.level  # prediction level, leaf or basic-level
tested_split = args.split
n_instances = args.instances
keep_stop_word = True
stop_word_as_anchor = False
summary_date = datetime.datetime.now().strftime("%m%d%Y")
load_model_file = f"./models-saved/cobweb/cobweb-seed{seed}-window{window}-leastfreq{least_freq}-instances{n_instances}-split{split}.json"
output_file = f"./test-summary/cobweb/test-summary-single_{pred_level}-split{tested_split}-{summary_date}.csv"


# Load existing model
tree = CobwebTree(0.000001, False, 0, True, False)
model_file = load_model_file
with open(model_file, 'r') as file:
	model_data = file.read()
tree.load_json(model_data)

""" test the external question set """

print("\nNow test with the challenge data. Prediction level used:", pred_level)

# Load the question along with answer options
print("\nGenerating question instances...")
question_instances = []
answer_options = []
df_questions = pd.read_csv('./data-te/testing_data.csv')
for i, row in df_questions.iterrows():
	question_text = row['question']
	options = {'a': row['a)'], 'b': row['b)'], 'c': row['c)'], 'd': row['d)'], 'e': row['e)']}
	options = lemmatized_options(options)
	question_instance = question2instance(question_text, window=window, test=True, keep_stop_word=keep_stop_word)
	question_instances.append(question_instance)
	answer_options.append(options)

# print("\nPreview on the question instances and options.")
# print(question_instances[:5])
# print(answer_options[:5])

def predict_option(probs_pred):
	available_options = [(index, option) for index, option in answer_options[i].items() if option in probs_pred['anchor'].keys()]
	if len(available_options) > 1:
		option_pred = sorted([(probs_pred['anchor'][option], random(), index, option) for index, option in available_options], reverse=True)[0][2]
	elif len(available_options) == 1:
		option_pred = available_options[0][0]
	else:
		option_pred = 'NA'
	return option_pred

# Generate answers and test the questions
df_answers = pd.read_csv('./data-te/test_answer.csv')
answers = df_answers['answer'].tolist()
test_leaf = False
test_basic = False
if pred_level in ["leaf", "both"]:
	test_leaf = True
if pred_level in ["basic", "both"]:
	test_basic = True

if test_leaf:
	predictions_leaf = []
	n_correct_leaf = 0
	for i in tqdm(range(len(question_instances))):
		node_leaf = tree.categorize(question_instances[i])
		option_pred = predict_option(node_leaf.predict_probs())
		predictions_leaf.append(option_pred)
		if option_pred == answers[i]:
			n_correct_leaf += 1
	print(f"\nThe accuracy on the challenge data (Prediction level: leaf): {n_correct / len(answers)} ({n_correct}/{len(answers)})")

if test_basic:
	predictions_basic = []
	n_correct_basic = 0
	for i in tqdm(range(len(question_instances))):
		node_leaf = tree.categorize(question_instances[i])
		option_pred = predict_option(node_leaf.get_basic_level().predict_probs())
		predictions_basic.append(option_pred)
		if option_pred == answers[i]:
			n_correct_basic += 1
	print(f"\nThe accuracy on the challenge data (Prediction level: basic): {n_correct / len(answers)} ({n_correct}/{len(answers)})")

# Write the summary back to the summary data.
df_summary = pd.read_csv("./data-te/testing_data.csv")
if test_leaf:
	df_summary['predictions_leaf'] = predictions_leaf
if test_basic:
	df_summary['predictions_basic'] = predictions_basic
df_summary['answer'] = answers
df_summary.to_csv(output_file, index=False)

