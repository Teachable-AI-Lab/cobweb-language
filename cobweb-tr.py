import os
import json
import random as rd
from random import random, shuffle
from collections import Counter
from tqdm import tqdm
import math
from multiprocessing import Pool
import pandas as pd
from utility import process_text, load_texts, story2instances, question2instance, lemmatized_options

from cobweb.cobweb import CobwebTree
from cobweb.visualize import visualize

if __name__ == "__main__":

	# Preprocess configuration:
	json_name = "holmes_stories.json"
	verbose = True
	limit = None
	window = 10  # the size of the "gram" (so context words = 2 * window)
	least_frequency = 3  # used to filter out the words having frequency less than some specified one.
	seed = 123  # random seed for shuffling instances
	keep_stop_word = True
	stop_word_as_anchor = False
	dummy = True
	if dummy:
		json_name = "holmes_stories_dummy.json"
	preview = False  # to preview the generated tokens and preprocessed instances

	# Trained instances configuration:
	# First specify the number of instances considered. If None, then all generated instances will be considered
	start_instance_id = None 
	end_instance_id = None
	# Then specify the size of training set by specifying the training portion
	train_portion = 1
	have_test_portion = False  # whether to introduce test set or not. If do so, the test set will be generated with the rest instances that are not in the training set
	# Finaly specify the number of training splits generated in the training set
	# Priority order: n_tr_splits > split_size
	n_tr_splits = 1
	split_size = 833333

	# Training process configuration:
	start_split = 0
	load_model = False
	# load_model_file = f"./model/cobweb-seed123-window{window}-leastfreq{least_frequency}-instances416667-split{start_split-1}.json"
	load_model_file = "./model/cobweb-seed123-window10-leastfreq3-instances5000000-split0.json"


	""" Load and preprocess the text data used """
	if verbose:
		print("Start the loading and preprocessing process.")

	# Create the integrated text from several text file (stories):
	if not os.path.isfile(json_name):
		if verbose:
			print("\nReading and preprocessing...")
		if dummy:
			stories_dir = "./data-tr-dummy"
		else:
			stories_dir = "./data-tr"
		stories = list(load_texts(stories_dir, limit=limit, keep_stop_word=keep_stop_word))
		with open(json_name, "w") as fout:
			json.dump(stories, fout, indent=4)
	else:
		if verbose:
			print("\nLoading the preprocessed stories...")
		with open(json_name, "r") as fin:
			stories = json.load(fin)

	overall_freq = Counter([word for story in stories for word in story])
	if preview:
		# A 200-first-word preview of some preprocessed story:
		print("\nPreview of the 200 first words of the first preprocessed story:")
		print(stories[0][:200])
		# You may see the overall word frequencies (50 most frequent words):
		print("\npreview of the 100 most frequent words frequency:")
		print(overall_freq.most_common(100))


	""" Generate the instances for Cobweb learning, the 'n-grams' """
	# Filter out the words having frequency >= least_frequency only:
	stories = [[word for word in story if overall_freq[word] >= least_frequency] for story in stories]


	# Generate instances (along with their story and anchor indices):
	print("\nNow generate the instances:")
	instances = []
	with Pool() as pool:
		processed_stories = pool.starmap(story2instances, [(story, window, stop_word_as_anchor) for story in stories])
		# for story_idx, story_instances in enumerate(processed_stories):
		for story_instances in tqdm(processed_stories):
			for anchor_idx, instance in story_instances:
				# instances.append((instance, story_idx, anchor_idx))
				instances.append(instance)

	rd.seed(seed)
	shuffle(instances)

	print("\nThe number of instances: {}".format(len(instances)))  # 16645730 (when least_frequency=3 and window=10)


	"""
	Generate training and test sets (within the Holmes data - not the external test set!)
	We generate n_tr_splits training splits + test split.
	After training all training splits, we store the model once.
	Then use it to test with the test split, and after that, train the additional test instances.
	Store the model another time.
	"""
	# if n_instances:
	# 	instances = instances[:n_instances]
	# if start_instance_id:
	# 	instances = instances[start_instance_id:]
	start_id = start_instance_id
	end_id =  end_instance_id
	instances = instances[start_id:end_id]
	n_instances = len(instances)
	tr_size = round(n_instances * train_portion)
	if n_tr_splits is not None:
		tr_split_size = round(tr_size / n_tr_splits)
	else:
		tr_split_size = split_size
		n_tr_splits = math.ceil(tr_size / tr_split_size)
	instances_splits = []
	for i in range(n_tr_splits):
		if i != n_tr_splits - 1:
			instances_splits.append(instances[i*tr_split_size:(i+1)*tr_split_size])
		else:
			instances_splits.append(instances[i*tr_split_size:tr_size])
	# instances_splits.append(instances[tr_size:])
	if have_test_portion:
		instances_te = instances[tr_size:]
		# print(instances_te[:5])
		instances_te_no_anchor = [{'context-before': instance['context-before'], 'context-after': instance['context-after']} for instance in instances_te]
	print(f"\n Have test set? {have_test_portion}")
	print(f"Here consider instance {start_instance_id} to instance {end_instance_id}.")
	print(f"There are {len(instances_splits)} training sets in total, and their sizes are {[len(split) for split in instances_splits]}.")
	if have_test_portion:
		print(f"There are {len(instances_te)} instances in the test set.")


	""" Model initialization """
	print(f"\nModel initialization: Load model - {load_model}.")
	tree = CobwebTree(0.000001, False, 0, True, False)
	if load_model:
		with open(load_model_file, 'r') as file:
			model_data = file.read()
		tree.load_json(model_data)


	""" Train Cobweb with the training splits """
	print("\nStart Training process.")
	for i in range(start_split, len(instances_splits)):
		print(f"\nNow train split {i + 1} / {len(instances_splits)}.")
		print(f"The number of instances: {len(instances_splits[i])}")
		for instance in tqdm(instances_splits[i]):
			tree.ifit(instance)
		# After training, store the model (json file):
		json_output = tree.dump_json()
		file_name = f"./models-saved/cobweb/cobweb-seed{seed}-window{window}-leastfreq{least_frequency}-instances{len(instances_splits[i])}-split{i}.json"
		print("Now save the model checkpoint:")
		with open(file_name, 'w') as json_file:
			json_file.write(json_output)


	""" Test the Cobweb trained with all training splits """
	# print("\nNow all the train splits are used. Test with the test data.")
	# anchors_te = [list(instance['anchor'].keys())[0] for instance in instances_te]
	# n_correct = 0
	# for i in tqdm(range(len(instances_te_no_anchor))):
	# 	probs_pred = tree.predict_probs(instances_te_no_anchor[i], nodes_pred, False, False)
	# 	anchor_pred = sorted([(probs_pred["anchor"][word], random(), word) for word in probs_pred['anchor']], reverse=True)[0][2]
	# 	if anchor_pred == anchors_te[i]:
	# 		n_correct += 1
	# test_acc = n_correct / len(instances_te)
	# print(f"Accuracy on the test data within the raw data: {test_acc} ({n_correct}/{len(instances_te)}).")


	""" Train Cobweb with the additional test set """
	if have_test_portion:
		print("\nNow train with the additional test set.")
		for instance in tqdm(instances_te):
			tree.ifit(instance)
		# After training, store the model (json file):
		json_output = tree.dump_json()
		file_name = f"./models-saved/cobweb/cobweb-seed{seed}-window{window}-leastfreq{least_frequency}-all-instances.json"
		with open(file_name, 'w') as json_file:
			json_file.write(json_output)

	print("\nTraining process is done!")

