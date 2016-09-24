import re
import cv2
import sys
import subprocess
import numpy as np
import scipy.io as sio
import json
import pdb
import pickle

def loadTokens(filename):
	tokens = json.load(open(filename))
	return [str(token) for token in tokens]

def main():
	image_dir = 'data/Images_vg/'
	print 'load image to regions'
	image2regions = json.load(open('image2regions.json'))
	print 'load test images index'
	images = json.load(open('./info/densecap_splits.json'))
	images = images['val']
	ims = int(sys.argv[1])
	ime = int(sys.argv[2])
	gpu = sys.argv[3]
	batch = '-1'#sys.argv[4]
	use_generate = True
	region2bbs = {}
	print 'start eval'
	for index, img in enumerate(images[ims: ime]):
		print 'progress: [%d/%d]' % (index + 1, ime - ims)
		image_path = image_dir + str(img) + '.jpg'
		gt_desc_file = 'gt_desc_%s.json' % batch
		if use_generate:
			descriptions = []
			reg_ids = []
			gt_boxes = []
			for reg in image2regions[str(img).decode('utf-8')]:
				desc = reg[u'phrase']
				desc = re.sub(r'[^\w\s]', '', desc)
				desc = desc.strip()
				desc = re.sub(r'[^\x00-\x7f]', r'', desc)
				desc = re.sub(' +', ' ', desc)
				desc = desc.lower()
				if reg[u'width'] * reg[u'height'] <= 400:
					continue
				descriptions.append(desc)
				gt_boxes.append([float(reg[u'x']), float(reg[u'y']), float(reg[u'width']), float(reg[u'height'])])
				reg_ids.append(reg[u'region_id'])
			json.dump(descriptions, open('gt_desc_%s.json' % batch, 'wb'))
			temp = {'x': np.transpose(np.array(gt_boxes))}
			sio.savemat('gt_boxes_%s.mat' % batch, temp)
			subprocess.call('th run_model.lua -checkpoint ../densecap/dense_paper.t7 \
			-gtDescFile %s -input_image %s -gpu %s -generateGTCaption %s -output_index %s' % (gt_desc_file, image_path, gpu, 'y', batch), shell = True)
			gen_gt_desc = json.load(open('gen_gt_desc.json'))
			json.dump(gen_gt_desc[0: len(reg_ids)], open('gen_gt_desc.json', 'wb'))
			gt_desc_file = 'gen_gt_desc.json'
		#continue normal test process
		descriptions = []
		reg_ids = []
		gt_boxes = []
		for reg in image2regions[str(img).decode('utf-8')]:
			desc = reg[u'phrase']
			desc = re.sub(r'[^\w\s]', '', desc)
			desc = desc.strip()
			desc = re.sub(r'[^\x00-\x7f]', r'', desc)
			desc = re.sub(' +', ' ', desc)
			desc = desc.lower()
			if reg[u'width'] * reg[u'height'] <= 400:
				continue
			descriptions.append(desc)
			gt_boxes.append([float(reg[u'x']), float(reg[u'y']), float(reg[u'width']), float(reg[u'height'])])
			reg_ids.append(reg[u'region_id'])
		json.dump(descriptions, open('gt_desc_%s.json' % batch, 'wb'))
		temp = {'x': np.transpose(np.array(gt_boxes))}
		sio.savemat('gt_boxes_%s.mat' % batch, temp)
		subprocess.call('th run_model.lua -checkpoint ../densecap/dense_paper.t7 -gtDescFile %s -input_image %s -gpu %s -output_index %s' % (gt_desc_file, image_path, gpu, batch), shell = True)
		bbs = sio.loadmat('bbs_%s.mat' % batch)['x']
		bbs = np.transpose(bbs)
		retrival_scores = sio.loadmat('retrival_scores_%s.mat' % batch)['x']
		subprocess.call('rm bbs_%s.mat' % batch, shell = True)
		subprocess.call('rm retrival_scores_%s.mat' % batch, shell = True)
		subprocess.call('rm gen_gt_desc.json', shell = True)
		retrival_scores = np.transpose(retrival_scores)
		im = cv2.imread(image_path)
		for idx, reg in enumerate(reg_ids):
			temp = []
			for i, bb in enumerate(bbs):
				revise_bb = [int(cor + 0.5) for cor in bb]
				revise_bb[2] = min(revise_bb[2], im.shape[1] - revise_bb[0])
				revise_bb[3] = min(revise_bb[3], im.shape[0] - revise_bb[1])
				temp.append(revise_bb + [retrival_scores[i, idx].tolist()])
			region2bbs[reg] = temp
		if index % 300 == 0:
			print 'save to file'
			pickle.dump(region2bbs, open('paper_dense_pretrained_scores_image_retrival_%s_to_%s.pkl' % (ims, ime), 'wb'))
			
	pickle.dump(region2bbs, open('paper_dense_pretrained_scores_image_retrival_%s_to_%s.pkl' % (ims, ime), 'wb'))
		
if __name__ == "__main__":
	main()
