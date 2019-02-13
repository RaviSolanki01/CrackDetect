import glob
import argparse
from glob import glob
from os import getcwd, chdir
from random import shuffle

parser = argparse.ArgumentParser()

parser.add_argument('--pos', dest = 'pos', default = None, type = str)
parser.add_argument('--neg', dest = 'neg', default = None, type = str)

args = parser.parse_args()

Positive = args.pos
Negative = args.neg

pos_img_list = []
neg_img_list = []

extension = "jpg"

with open (Positive,"r") as txt:
	for single_path in txt.readlines():
		single_path = single_path[:-1]
		
		saved = getcwd()
		chdir(single_path)
		local_pos = glob('*.' + extension)
		complete_pos = [single_path+ "/" + e for e in local_pos]

		for imag in complete_pos:
			pos_img_list.append(imag)

		chdir(saved)

with open (Negative,"r") as txt2:
	for single_npath in txt2.readlines():
		single_npath = single_npath[:-1]
		saved = getcwd()

		chdir(single_npath)
		local_npos = glob('*.' + extension)
		complete_npos = [single_npath+ "/" + e for e in local_npos]

		for imag2 in complete_npos:
			neg_img_list.append(imag2)

		chdir(saved)

shuffle(pos_img_list)
shuffle(neg_img_list)

pos_test_size = len(pos_img_list)//5
pos_test_data = pos_img_list[:pos_test_size]
pos_train_data = pos_img_list[pos_test_size:]

neg_test_size = len(neg_img_list)//5
neg_test_data = neg_img_list[:neg_test_size]
neg_train_data = neg_img_list[neg_test_size:]

with open('positive_training.txt', 'w') as f:
    for item in pos_train_data:
        f.write("%s\n" % item)

with open('positive_test.txt', 'w') as f:
    for item in pos_test_data:
        f.write("%s\n" % item)

with open('negative_training.txt', 'w') as f:
    for item in neg_train_data:
        f.write("%s\n" % item)

with open('negative_test.txt', 'w') as f:
    for item in neg_test_data:
        f.write("%s\n" % item)