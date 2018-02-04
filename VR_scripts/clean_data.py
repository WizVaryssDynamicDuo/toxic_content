
# Timestamp of Creation : 2018-Jan-07 21:50
# Author : Varun Rawal
# Title : Cleaning of Datasets

import re
import pickle

def stringRepresentsInt(s):
	try: 
		int(s)
		return True
	except ValueError:
		return False


def endOfSample(commaSplitList):
	if(len(commaSplitList)<7):
		return False
	for label in commaSplitList[-6:]:
		if not stringRepresentsInt(label):
			return False
		else:
			# label is of integer type
			int_label = int(label)
			if int_label!=0 and int_label!=1:
				return False
	return True

def updateIntegerLabels(commaSplitList, record):
	idx = 0
	labelTypeMap = ["toxic","severe_toxic","obscene","threat","insult","identity_hate"]
	for label in commaSplitList[-6:]:
		int_label = int(label)
		record[labelTypeMap[idx]] = int_label
		idx+=1
	return record


def trainDataParser(filename):
	sampleBeginFlag = True
	sample_record = {}
	comment_id = -1
	comment_text = ""

	with open(filename) as f:
		allLines = f.readlines()
		totalCount = len(allLines)
		count=0
		oldPC = -1
		# you may also want to remove whitespace characters like `\n` at the end of each line
		lineContent = [x.strip() for x in allLines]
		for nextLine in lineContent:
			count+=1;
			percentCompletion = int(count/totalCount*100.0)
			if percentCompletion%20==0 and percentCompletion!=oldPC:
				print("Completed processing ", percentCompletion, "% Lines \n")
				oldPC = percentCompletion
			commaSplitList = nextLine.split(',')
			sampleEndFlag = endOfSample(commaSplitList)

			if sampleBeginFlag:
				comment_text_beg = 1
				#extract the id from list[0]
				string_id = commaSplitList[0]
				if stringRepresentsInt(string_id):
					comment_id = int(string_id)
					sample_record['id'] = comment_id
				else:
					# error
					#cannot parse this line as it is corrupt (possibly the header / footer lines)
					sample_record = {}
					sampleBeginFlag = True
					comment_text = ""
					continue
			else:
				comment_text_beg = 0

			comment_text += ' '.join(commaSplitList[comment_text_beg:-6]) if sampleEndFlag else ' '.join(commaSplitList[comment_text_beg:])

			if sampleEndFlag:
				sample_record['comment_text'] = comment_text
				sample_record = updateIntegerLabels(commaSplitList, sample_record)
				train_records.append(sample_record)
				#refresh all variables for next record
				sample_record = {}
				sampleBeginFlag = True
				comment_text = ""
			else:
				sampleBeginFlag = False


train_records = []

path_2_traindata = "../../../Datasets/train.csv"
path_2_testdata = "../../../Datasets/test.csv"
path_2_trainPickle = "../../../Datasets/trainDataRecords.pickle"

trainDataParser(path_2_traindata)

print("Retrieved ", len(train_records), "training data sample records from ", path_2_traindata, "! Serializing the records to ", path_2_trainPickle, " ... ")

with open(path_2_trainPickle, "wb") as f:
	pickle.dump(train_records, f)
