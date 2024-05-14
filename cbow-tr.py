import os
import json
import random as rd
from random import random, shuffle
from collections import Counter
from tqdm import tqdm
import math
from multiprocessing import Pool
import pandas as pd
import torch
from utility import process_text, load_texts, story2instances, question2instance, lemmatized_options

from models.cbow_pytorch import CbowModel


def instance2tokens(instance):
	anchor = list(instance['anchor'].keys())[0]
	context_before = sorted([(prob, random(), word) for word, prob in instance['context-before'].items()])
	context_before = [word for (_, _, word) in context_before]
	context_after = sorted([(prob, random(), word) for word, prob in instance['context-after'].items()], reverse=True)
	context_after = [word for (_, _, word) in context_after]
	context = tuple(context_before + context_after)
	return context, anchor


if __name__ == "__main__":

	# Preprocess configuration:
	json_name = "holmes_stories.json"  # the json file storing the parsed tokens of the stories
	verbose = True
	limit = None
	window = 10  # the size of the "gram" (so # of context words = 2 * window)
	least_frequency = 3  # used to filter out the words having frequency less than some specified one.
	seed = 123  # random seed for shuffling instances
	keep_stop_word = True
	stop_word_as_anchor = False
	max_vocab_size = 20000000
	dummy = True
	if dummy:
		json_name = "holmes_stories_dummy.json"
	
	# Model training configuration:
	embedding_dim = 100
	batch_size = 64
	epochs = 3
	use_cuda = False

	# Trained instances configuration:
	# First specify the number of instances considered. If None, then all generated instances will be considered
	start_instance_id = None
	end_instance_id = None
	# Then specify the size of training set by specifying the training portion
	train_portion = 1  # the portion of the generated instances being the training set
	have_test_portion = False  # whether to introduce test set or not. If do so, the test set will be generated with the rest instances that are not in the training set
	# Finaly specify the number of training splits generated in the training set
	# Priority order: n_tr_splits > split_size
	n_tr_splits = 1  # nbr of training splits
	split_size = 833333

	# Training process configuration:
	start_split = 0
	start_instance_id = None  # start considering the instance with some speific id
	load_model = False
	load_model_file = f"./save-models/chow-seed{seed}-window{window}-leastfreq{least_frequency}-allinstances-split{start_split}.pth"

	if verbose:
		print("Start the loading and preprocessing process.")

	# Create the integrated text from several text file (stories):
	# If the parsed token file exists, just load the file
	# Otherwise generate the parsed token JSON file
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

	""" Generate the instances for Cobweb learning, the 'n-grams' """
	# Filter out the words having frequency >= least_frequency only:
	overall_freq = Counter([word for story in stories for word in story])
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


	# Dataset splits:
	start_id = start_instance_id
	end_id =  end_instance_id
	instances = instances[start_id:end_id]
	# if n_instances:
	# 	instances = instances[:n_instances]
	# if start_instance_id:
	# 	instances = instances[start_instance_id:]
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


	# Model initialization:
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if use_cuda else torch.device("cpu")
	print(f"\nDevice used: {device}")
	if load_model:
		model = torch.load(load_model_file)
	else:
		model = CbowModel(max_vocab_size=max_vocab_size, device=device, embedding_dim=embedding_dim, batch_size=batch_size, window=window, epochs=epochs)


	for i in range(start_split, len(instances_splits)):
		print(f"\nNow train split {i + 1} / {len(instances_splits)}.")
		# instances -> tokens
		data = [instance2tokens(instance) for instance in instances_splits[i]]
		contexts = [data[j][0] for j in range(len(data))]
		anchors = [data[j][1] for j in range(len(data))]

		# Training:
		# for j in tqdm(range(len(data))):
		# 	model.train(data[j][0], data[j][1])
		model.train(contexts, anchors)

		# Save model
		torch.save(model, f'./save-models/cbow/cbow-seed{seed}-window{window}-leastfreq{least_frequency}-instances{len(anchors)}-split{i}-device{device}.pth')

	# # Training:
	# for j in tqdm(range(len(data))):
	# 	model.train(data[j][0], data[j][1])

	# # Testing:
	# print(model.predict(('damn', 'bad', 'oh')))

	# # Save model?
	# # torch.save(model, f'./model/chow-seed{seed}-window{window}-leastfreq{least_frequency}-instances{len(instances_splits[i])}-split{i}.pth')
	# torch.save(model, './model/chow_model_test.pth')










