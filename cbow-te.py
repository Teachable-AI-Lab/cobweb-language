import os
import json
from random import random, shuffle
from tqdm import tqdm
import spacy
import torch
import logging
from utility import process_text, load_texts, story2instances, question2instance, lemmatized_options
import pandas as pd
import numpy as np
import datetime

# Configuration:
keep_stop_word = True  # Keep stop words in the question instances or not
split_id = 0  # The id of the last trained split of the loaded model
summary_date = datetime.datetime.now().strftime("%m%d%Y")
output_summary_file = f"./test-summary/cbow/test-summary-cbow-pytorch-firstsplit-split{split_id}-{summary_date}.csv"
preview_questions = False

# Set up logging:
# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# Load model
cuda = False  # use GPUs?
load_model_file = f"./models-saved/cbow/cbow-seed123-window10-leastfreq3-instances416667-split{split_id}.pth"
if cuda:
	model = torch.load(load_model_file)
else:
	model = torch.load(load_model_file, map_location=torch.device('cpu'))
	model.device = torch.device("cpu")


# Load the question along with answer options
questions = []
answer_options = []
df_questions = pd.read_csv('./data-te/testing_data.csv')
for i, row in df_questions.iterrows():
	question_tokens = process_text(row['question'], test=True, keep_stop_word=keep_stop_word)
	options = {'a': row['a)'], 'b': row['b)'], 'c': row['c)'], 'd': row['d)'], 'e': row['e)']}
	options = lemmatized_options(options)
	questions.append(question_tokens)
	answer_options.append(options)
if preview_questions:
	print("\nPreview on the questions.")
	print(questions[:5])
	print("\nPreview on the answer options.")
	print(answer_options[:5])


# Generate answers and test the questions
df_answers = pd.read_csv('./data-te/test_answer.csv')
answers = df_answers['answer'].tolist()
predictions = []
n_correct = 0
for i in tqdm(range(len(questions))):

	probs_pred = model.predict(questions[i])
	available_options = [(index, option) for index, option in answer_options[i].items() if option in probs_pred.keys()]
	if len(available_options) > 1:
		option_pred = sorted([(probs_pred[option], random(), index, option) for index, option in available_options], reverse=True)[0][2]
	elif len(available_options) == 1:
		option_pred = available_options[0][0]
	else:
		option_pred = 'NA'
	predictions.append(option_pred)
	if option_pred == answers[i]:
		n_correct += 1
print(f"\nThe accuracy on the challenge data is {n_correct / len(answers)} ({n_correct}/{len(answers)}).")

# Write the summary back to the summary data.
df_summary = pd.read_csv("./data-te/testing_data.csv")
df_summary['predictions'] = predictions
df_summary['answer'] = answers
df_summary.to_csv(output_summary_file, index=False)
