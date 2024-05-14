import re
import os
import spacy
from multiprocessing import Pool
from spacy.tokenizer import Tokenizer

# Import the sentence processor from spacy
nlp = spacy.load("en_core_web_sm", disable = ['parser'])
nlp.add_pipe("sentencizer")
nlp.max_length = float('inf')

# Nishant's processor:
# nlp = spacy.blank("en")
# nlp.max_length = float('inf')
# infixes = tuple([r"\w+'s\b", r"\w+'t\b", r"\d+,\d+\b", r"(?<!\d)\.(?!\d)"] +  nlp.Defaults.prefixes)
# infix_re = spacy.util.compile_infix_regex(infixes)
# nlp.tokenizer = Tokenizer(nlp.vocab, infix_finditer=infix_re.finditer)

# Test:
# tokens = nlp("I ate Jack's lunch this morning, and it costed me 1,000 dollars - Jeez! I can't make it! 15.5?")
# for token in tokens:
# 	print(token)
# text = [token.lemma_.lower() for token in tokens if (not token.is_punct)]
# print(text)
# text = [token.text.lower() for token in tokens if (not token.is_punct)]
# print(text)
# text = [token.lemma_.lower() for token in tokens if (not token.is_punct and not token.is_stop)]
# print(text)


""" The following functions are used for preprocessing in loading the stories """

def process_text(text, test=False, keep_stop_word=False):
	""" Load and preprocess a single (line) of text """
	# Preprocess
	if test:
		# punc = re.compile(r"[^_a-zA-Z,.!?:;\s]")
		punc = re.compile(r"[^_a-zA-Z0-9,.!?:;\s]")
	else:
		# punc = re.compile(r"[^a-zA-Z,.!?:;\s]")
		punc = re.compile(r"[^a-zA-Z0-9,.!?:;\s]")
	whitespace = re.compile(r"\s+")

	# Nishant:
	text = re.sub(r'--', r' ', text)

	text = punc.sub("", text)
	text = whitespace.sub(" ", text)
	text = text.strip().lower()
	
	# Parse
	text = nlp(text)
	if keep_stop_word:
		text = [token.lemma_.lower() for token in text if (not token.is_punct)]
	else:
		text = [token.lemma_.lower() for token in text if (not token.is_punct and not token.is_stop)]
	return text


def process_file(idx, name, fp, verbose=True, keep_stop_word=False):
	""" Load and preprocess a text file """
	if verbose:
		print("Processing file {} - {}".format(idx, name))
	if not re.search(r'^[A-Z0-9]*.TXT$', name):
		return None
	with open(fp, 'r', encoding='latin-1') as fin:
		skip = True
		text = ""
		for line in fin:
			if not skip and not "project gutenberg" in line.lower():
				text += line
			elif "*END*THE SMALL PRINT! FOR PUBLIC DOMAIN ETEXTS" in line:
				skip = False

		output = process_text(text, keep_stop_word=keep_stop_word)
		return output


def load_texts(training_dir, limit=None, keep_stop_word=False):
	""" Load the preprocessed texts used for training """
	for path, subdirs, files in os.walk(training_dir):
		if limit is None:
			limit = len(files)
		texts = [(idx, name, os.path.join(path, name), True, keep_stop_word) for idx, name in enumerate(files[:limit])]

		# Preprocess the text files in parallel
		with Pool() as pool:
			outputs = pool.starmap(process_file, texts)
			for output in outputs:
				if output is None:
					continue
				yield output




""" The following functions are used for the instances (n-grams) generation: anchor + context words """

def old_get_instance(text, anchor_idx, anchor_wd, window):
	""" Generate an instance {'anchor': {anchor_word: 1}, 'context': {context_1: ..., context_2: ..., ...}} """
	before_text = text[max(0, anchor_idx - window):anchor_idx]
	after_text = text[anchor_idx + 1:anchor_idx + 1 + window]
	ctx_text = before_text + after_text
	ctx = {}

	# In a language task, the context words are not considered as simple counts.
	# Considering the proximity to the anchor word, the further the context word to the anchor, the less weight it will have
	for i, w in enumerate(before_text):
		ctx[w] = 1 / abs(len(before_text) - i)
	for i, w in enumerate(after_text):
		ctx[w] = 1 / (i + 1)

	instance = {}
	instance['context'] = ctx
	if anchor_wd is None:
		return instance
	instance['anchor'] = {anchor_wd: 1}
	return instance


def get_instance(text, anchor_idx, anchor_wd, window):
	""" Generate an instance {'anchor': {anchor_word: 1}, 'context': {context_1: ..., context_2: ..., ...}} """
	""" Introduce context words before and after the anchor word."""

	before_text = text[max(0, anchor_idx - window):anchor_idx]
	after_text = text[anchor_idx + 1:min(anchor_idx + 1 + window, len(text))]
	context_before = {}
	context_after = {}

	# In a language task, the context words are not considered as simple counts.
	# Considering the proximity to the anchor word, the further the context word to the anchor, the less weight it will have
	for i, w in enumerate(before_text):
		context_before[w] = 1 / abs(len(before_text) - i)
	for i, w in enumerate(after_text):
		context_after[w] = 1 / (i + 1)

	instance = {}
	instance['context-before'] = context_before
	instance['context-after'] = context_after
	if anchor_wd is None:
		return instance
	instance['anchor'] = {anchor_wd: 1}
	return instance


def question2instance(text, window, test=False, keep_stop_word=False):
	""" Load and preprocess a single (line) of text """
	# Preprocess
	instance = {}
	if test:
		# punc = re.compile(r"[^_a-zA-Z,.!?:;\s]")
		punc = re.compile(r"[^_a-zA-Z0-9,.!?:;\s]")
	else:
		# punc = re.compile(r"[^a-zA-Z,.!?:;\s]")
		punc = re.compile(r"[^a-zA-Z0-9,.!?:;\s]")
	whitespace = re.compile(r"\s+")
	text = punc.sub("", text)
	text = whitespace.sub(" ", text)
	text = text.strip().lower()
	
	# Parse
	text = nlp(text)
	anchor_id = 0
	# find the anchor id
	for i in range(len(text)):
		if "_" in text[i].text:
			anchor_id = i
			break
	if keep_stop_word:
		before_anchor = [token.lemma_.lower() for token in text[max(0, anchor_id - window):anchor_id] if (not token.is_punct)]
		after_anchor = [token.lemma_.lower() for token in text[anchor_id + 1:min(len(text), anchor_id + 1 + window)] if (not token.is_punct)]
	else:
		before_anchor = [token.lemma_.lower() for token in text[max(0, anchor_id - window):anchor_id] if (not token.is_punct and not token.is_stop)]
		after_anchor = [token.lemma_.lower() for token in text[anchor_id + 1:min(len(text), anchor_id + 1 + window)] if (not token.is_punct and not token.is_stop)]
	context_before = {}
	context_after = {}
	for i, w in enumerate(before_anchor):
		context_before[w] = 1 / abs(len(before_anchor) - i)
	for i, w in enumerate(after_anchor):
		context_after[w] = 1 / (i + 1)
	instance['context-before'] = context_before
	instance['context-after'] = context_after
	return instance


def lemmatized_options(option_instance):
	instance = {}
	for k, v in option_instance.items():
		v = nlp(v)
		instance[k] = [token.lemma_.lower() for token in v][0]
	return instance


def _story2instances(story, window, stop_word_as_anchor=False):
	for anchor_idx, anchor_wd in enumerate(story):
		if stop_word_as_anchor:
			yield anchor_idx, get_instance(story, anchor_idx, anchor_wd, window=window)
		else:
			if anchor_wd not in nlp.Defaults.stop_words:
				yield anchor_idx, get_instance(story, anchor_idx, anchor_wd, window=window)


def story2instances(story, window, stop_word_as_anchor=False):
	return list(_story2instances(story, window, stop_word_as_anchor=stop_word_as_anchor))






