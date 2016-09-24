import json
import pickle
import pdb
import numpy as np
import cv2
from rankaccuracy  import IoU, nms
import sys

def drawGT(img, region):
	cv2.rectangle(img, (int(region[u'x']), int(region[u'y'])), \
				  (int(region[u'x'] + region[u'width']), int(region[u'y'] + region[u'height'])), (0, 255, 0), 4)

def drawRetrival(img, region, bbs, K, model = 'cnn'):
	crop = [region[u'x'], region[u'y'], region[u'width'], region[u'height']]
	bbs = nms(bbs, 0.3, 4, high_good = (model == 'dense'))
	good = 0 
	for bb in bbs[0: K]:
		if IoU(crop, bb) >= 0.1:
			good += 1
		cv2.rectangle(img, (int(bb[0]), int(bb[1])), \
					(int(bb[0] + bb[2]), int(bb[1] + bb[3])), (255, 0, 0), 2)
		cv2.putText(img, str(bb[4]), (int(bb[0]), int(bb[1])), 0, 0.3, (255, 255, 255), 1)
	return good

def drawIoU(img, region, bbs, K):
	crop = [region[u'x'], region[u'y'], region[u'width'], region[u'height']]
	for bb in bbs:
		bb.append(IoU(crop, bb))
	bbs = sorted(bbs, key = lambda tup: tup[-1], reverse = True)
	for bb in bbs[0: K]:
		cv2.rectangle(img, (int(bb[0]), int(bb[1])), \
					(int(bb[0] + bb[2]), int(bb[1] + bb[3])), (0, 0, 255), 2)

def main():
	#image2regions = json.load(open('image2regions.json'))
	print 'load in data'
	ims = sys.argv[1]
	ime = sys.argv[2]
	regions_dict = json.load(open('./data/regions_dict.json'))
    regions2bbs = pickle.load(open('./paper_dense_pretrained_scores_image_retrival_%s_to_%s.pkl' % (ims, ime)))
	print 'start visualization'
	for reg in regions2bbs.keys():
		image = regions_dict[str(reg).decode('utf-8')][u'image_id']
		img = cv2.imread('data/Images_vg/' + str(image) + '.jpg')
		drawIoU(img, regions_dict[str(reg).decode('utf-8')], regions2bbs[reg], 10)
		drawGT(img, regions_dict[str(reg).decode('utf-8')])
		good = drawRetrival(img, regions_dict[str(reg).decode('utf-8')], regions2bbs[reg], 10, model = model)
		cv2.putText(img, regions_dict[str(reg).decode('utf-8')][u'phrase'], \
		(int(regions_dict[str(reg).decode('utf-8')][u'x']), int(regions_dict[str(reg).decode('utf-8')][u'y'])), \
		0, 0.5, (0, 0, 0), 2)
		if good > 2:
			cv2.imwrite('./retrival_visualization/perfect_%d_%d.jpg' % (image, reg), img)
		elif good >= 1:
			cv2.imwrite('./retrival_visualization/good_%d_%d.jpg' % (image, reg), img)
		else:
			cv2.imwrite('./retrival_visualization/%d_%d.jpg' % (image, reg), img)


if __name__ == "__main__":
	main()

