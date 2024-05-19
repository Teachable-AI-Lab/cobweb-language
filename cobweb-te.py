from tqdm import tqdm
import pandas as pd
import argparse
from random import random
import datetime
from utility import question2instance, lemmatized_options
from cobweb.cobweb import CobwebTree

parser = argparse.ArgumentParser()
parser.add_argument('--nodes', type=int, default=1000, help='the number of expanded nodes')
parser.add_argument('--split', type=int, default=11, help='the tested split index, start with 0')
parser.add_argument('--instances', type=int, default=1300000, help='the number of instances trained in the loaded model')
args = parser.parse_args()

# Configuration:
verbose = True
window = 10  # the size of the "gram" (so context words = 2 * window)
seed = 123  # random seed for shuffling instances
least_freq = 3
nodes_pred = args.nodes  # number of nodes used in multiple prediction
tested_split = args.split
n_instances = args.instances
used_cores = 120  # nbr of cores used in parallel prediction
keep_stop_word = True
stop_word_as_anchor = False
summary_date = datetime.datetime.now().strftime("%m%d%Y")
default_answer = "c"
# load_model_file = f"./models-saved/cobweb/cobweb-seed123-window10-leastfreq3-instances416667-split{tested_split}.json"
load_model_file = f"./models-saved/cobweb/cobweb-seed{seed}-window{window}-leastfreq{least_freq}-instances{n_instances}-split{tested_split}.json"
output_file = f"./test-summary/cobweb/test-summary-node{nodes_pred}-split{tested_split}-{summary_date}-parallel.csv"

# Initialize tree:
tree = CobwebTree(0.000001, False, 0, True, False)
# Load existing model:
with open(load_model_file, 'r') as file:
    model_data = file.read()
tree.load_json(model_data)


""" test the external question set """

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

# Generate answers and test the questions
df_answers = pd.read_csv('./data-te/test_answer.csv')
answers = df_answers['answer'].tolist()
predictions = []
actual_predictions = []
actual_predictions_prob = []
actual_predictions_2 = []
actual_predictions_prob_2 = []
n_correct = 0
print(f"\nCobweb: {nodes_pred} expanded nodes, testing on checkpoint {tested_split}, {used_cores} used cores.")
probs_pred_all = tree.predict_probs_parallel(question_instances, nodes_pred, False, False, used_cores)
for i in tqdm(range(len(question_instances))):
    # probs_pred = tree.predict_probs(question_instances[i], nodes_pred, False, False)
    probs_pred = probs_pred_all[i]
    available_options = [(index, option) for index, option in answer_options[i].items() if option in probs_pred['anchor'].keys()]
    if len(available_options) > 1:
        option_pred = sorted([(probs_pred['anchor'][option], random(), index, option) for index, option in available_options], reverse=True)[0][2]
    elif len(available_options) == 1:
        option_pred = available_options[0][0]
    else:
        option_pred = default_answer
    predictions.append(option_pred)
    if option_pred == answers[i]:
        n_correct += 1

    # Additionally, what are the predicitons of Cobweb disregarding the answer options?
    # print(probs_pred)
    anchors_pred = sorted([(probs_pred['anchor'][anchor], random(), anchor) for anchor in probs_pred['anchor']], reverse=True)
    actual_predictions.append(anchors_pred[0][2])
    actual_predictions_prob.append(anchors_pred[0][0])
    actual_predictions_2.append(anchors_pred[1][2])
    actual_predictions_prob_2.append(anchors_pred[1][0])
    # anchors_pred = sorted([(probs_pred['anchor'][anchor], random(), anchor) for anchor in available_options], reverse=True)
print(f"\nThe accuracy on the challenge data is {n_correct / len(answers)} ({n_correct}/{len(answers)}).")

# Write the summary back to the summary data.
df_summary = pd.read_csv("./data-te/testing_data.csv")
df_summary['prediction'] = predictions
df_summary['answer'] = answers
df_summary['actual_pred'] = actual_predictions
df_summary['actual_pred_prob'] = actual_predictions_prob
df_summary.to_csv(output_file, index=False)

