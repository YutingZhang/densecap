import json
import pickle
import pdb
import numpy as np
import cv2
import sys

#crop format: [x, y, width, height]
def IoU(region_current, object_current):
    totalarea = object_current[2] * object_current[3] + region_current[2] * region_current[3]
    
    if region_current[0] <= object_current[0]:
        x_left = object_current[0]
    else:
        x_left = region_current[0]
    
    if region_current[1] <= object_current[1]:
        y_left = object_current[1]
    else:
        y_left = region_current[1]
    
    if region_current[0] + region_current[2] >= object_current[0] + object_current[2]:
        x_right = object_current[0] + object_current[2]
    else:
        x_right= region_current[0] + region_current[2]
    
    if region_current[1] + region_current[3] >= object_current[1] + object_current[3]:
        y_right = object_current[1] + object_current[3]
    else:
        y_right= region_current[1] + region_current[3]
    
    if x_right <= x_left:
        intersection = 0
    elif y_right <= y_left:
        intersection = 0
    else:
        intersection = (x_right - x_left) * (y_right - y_left)
	union = totalarea - intersection
	return 1.0 * intersection / union

#bbs_list should be a list of tuple, [(x, y, width, height, score1, ...)]
def nms(bbs_list, IoU_threshold, score_index, high_good = False):
	valid = np.ones(len(bbs_list))
	bbs_list = sorted(bbs_list, key = lambda tup: tup[score_index], reverse = high_good)
	for idx, bbs in enumerate(bbs_list):
		if valid[idx] == 0:
			continue
		for i in range(idx + 1, len(bbs_list)):
			if IoU(bbs, bbs_list[i]) > IoU_threshold:
				valid[i] = 0
	return [bbs_list[i] for i in range(len(bbs_list)) if valid[i] == 1]
def checkSame(crop1, crop2):
	same = True
	for i in range(4):
		if crop1[i] != crop2[i]:
			same = False
			break
	return same

#bbs should be processed by nms with IoU 0.3
def rankaccuracy(region, bbs, IoU_threshold, K):
	crop = [region[u'x'], region[u'y'], region[u'width'], region[u'height']]
	good = False 
	for bb in bbs[0: K]:
		if IoU(crop, bb) >= IoU_threshold and not checkSame(crop, bb):
			good = True
			break
	return good

def main():
	#image2regions = json.load(open('image2regions.json'))
	print 'load in data'
	regions_dict = json.load(open('data/regions_dict.json'))
	if sys.argv[3] == 'dense':
		regions2bbs = pickle.load(open('./paper_dense_pretrained_scores_image_retrival_%s_to_%s.pkl' % (sys.argv[1], sys.argv[2])))
	else:
		regions2bbs = pickle.load(open('./scores_image_retrival_small_meteor_dense_%s_%s.pkl' % (sys.argv[1], sys.argv[2])))
	print 'start analysis'
	correct = np.zeros([5, 4])
	row = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
	col = np.array([1, 2, 5, 10])
	for reg in regions2bbs.keys():
		bbs = regions2bbs[reg]
		bbs = nms(bbs, 0.3, 4, high_good = (sys.argv[3] == 'dense'))
		for ir, r in enumerate(row):
			for ic, c in enumerate(col):
				correct[ir, ic] += rankaccuracy(regions_dict[str(reg).decode('utf-8')], bbs, r, c)
	print correct / len(regions2bbs)

if __name__ == "__main__":
	main()

