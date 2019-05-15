import glob
import os
import sys
import csv
import gensim
import numpy as np

def recordResponse(filename, response, header,
				   separator = ',',
				   ender = "\n"):
	""" Generic file saving function that iteratively saves to same file line by line. Potentially slow with a lot of opening file. """
	if os.path.exists(filename):
		writeCode = 'a'
		with open(filename, writeCode) as f:
			record = ""
			for value in header:
				record += str(response[value]) + separator
			record = record[:-len(separator)]
			record += ender
			f.write(record)
	else:
		writeCode = 'w'
		with open(filename, writeCode) as f:
			record = ""
			for variable in header:
				record += variable + separator
			record = record[:-len(separator)]
			record += ender
			f.write(record)
			record = ""
			for value in header:
				record += str(response[value]) + separator
			record = record[:-len(separator)]
			record += ender
			f.write(record)

def getResponses(all_words_flag = False,
				 data_folder = '/Participant_Data',
				 filetype = '.csv'):
	""" Get the words produced by each participant. Participant data assumed to be in .csv file for each person. """
	participant_data = {}
	if all_words_flag:
		all_words = []
	for file in glob.glob(os.getcwd() + data_folder + "/*" + filetype):
		with open(file, 'r') as f:
			contents = csv.reader(f)
			participant_data[file] = list(contents)[0] # Set to only take the first column due to structure of my .csv files
			if all_words_flag:
				all_words = all_words + participant_data[file]
	# Other filetypes are easy to write. Use same general function, usually, and use f.readlines()
	if all_words_flag:
		all_words = list(set(all_words))
		return participant_data, all_words
	return participant_data

def getEmbeddings(model, words):
	""" Extracts word embeddings from a model. Words is a list that stores all words you want to extract. """
	embeddings = {}
	for word in words:
		embeddings[word] = model[word]
	return embeddings

def getMovement(embeddings,
				series):
	""" Returns difference from word1 to word2 for series of words in list form """
	movements = []
	for word1, word2 in zip(series[:-1], series[1:]):
		movements.append(embeddings[word2] - embeddings[word1])
	return movements

def getData(model, responses,
			all_words = None,
			output_file_all = "word2vec_Embeddings_All.csv",
			output_file_movement = "word2vec_Movement.csv"):
	""" Gets embeddings for each file, extracts embeddings for each response, gets the differences between those embeddings, and saves everything """
	for response_file in responses:
		response_embeddings = getEmbeddings(model, responses[response_file])
		response_movements = getMovement(response_embeddings, responses[response_file])
		saveMovement(response_file = response_file,
					 series = responses[response_file],
					 movements = response_movements,
					 output_file = output_file_movement)
	if all_words != None:
		all_embeddings = getEmbeddings(model, all_words)
		saveEmbeddings(embeddings = all_embeddings, 
					   output_file = output_file_all)

def saveMovement(response_file, series, movements, output_file,
				  header = ['response_file', 'series', 'numTransition', 'word1', 'word2']):
	""" Saves movement embeddings one-at-a-time to the output file. Iterates through embeddings along with each word in the list """
	for index, (word1, word2, movement) in enumerate(zip(series[:-1], series[1:], movements)):
		output = {'response_file': response_file,
				  'series': ' '.join(series),
				  'numTransition': index,
				  'word1': word1,
				  'word2': word2}
		header_addition = ['movement_{}'.format(x) for x in range(len(movement))]
		for value, key in zip(movement, header_addition):
			output[key] = value
		recordResponse(filename = output_file,
					   response = output,
					   header = (header + header_addition))

def saveEmbeddings(embeddings, output_file,
					header = ['word']):
	for word in embeddings:
		output = {'word': word}
		header_addition = ['dimension_{}'.format(x) for x in range(len(embeddings[word]))]
		for value, key in zip(embeddings[word], header_addition):
			output[key] = value
		recordResponse(filename = output_file,
					   response = output,
					   header = (header + header_addition))

if __name__ == '__main__':
	# Get the words that participants produced
	responses, all_words = getResponses(all_words_flag = True)
	# Get the model
	model = gensim.models.KeyedVectors.load_word2vec_format('./Semantic_Space/GoogleNews-vectors-negative300.bin', binary=True)
	# Get embeddings and movements and save movements
	getData(model = model, responses = responses, all_words = all_words)