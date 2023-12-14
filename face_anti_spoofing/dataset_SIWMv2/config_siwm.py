# -*- coding: utf-8 -*-
# Copyright 2022
# 
# Authors: Xiao Guo, Yaojie Liu, Anil Jain, and Xiaoming Liu.
# 
# All Rights Reserved.s
# 
# This research is based upon work supported by the Office of the Director of 
# National Intelligence (ODNI), Intelligence Advanced Research Projects Activity
# (IARPA), via IARPA R&D Contract No. 2017-17020200004. The views and 
# conclusions contained herein are those of the authors and should not be 
# interpreted as necessarily representing the official policies or endorsements,
# either expressed or implied, of the ODNI, IARPA, or the U.S. Government. The 
# U.S. Government is authorized to reproduce and distribute reprints for 
# Governmental purposes not withstanding any copyright annotation thereon. 
# ==============================================================================
import os
import abc
import csv
from glob import glob

class Config_siwm(object):
		
	LI_DATA_DIR = []
	SP_DATA_DIR = []
	LI_DATA_DIR_TEST = []
	SP_DATA_DIR_TEST = []
	spoof_type_dict = {'Co': 'Makeup_Co', 'Im': 'Makeup_Im', 'Ob': 'Makeup_Ob',
						'Half': 'Mask_Half', 'Mann': 'Mask_Mann', 'Paper': 'Mask_Paper',
						'Sil': 'Mask_Silicone', 'Trans': 'Mask_Trans', 'Print': 'Paper',
						'Eye': 'Partial_Eye', 'Funnyeye': 'Partial_Funnyeye',
						'Mouth': 'Partial_Mouth', 'Paperglass': 'Partial_Paperglass',
						'Replay': 'Replay'}

	def __init__(self, pro, unknown='None'):
		# Spoof type dictionary.

		"""
		the configuration class for siw-mv2 dataset.
		"""
		self.dataset = "SiWM-v2"
		self.spoof_img_root = '/home/kangcaixin/chenjiawei/siw-Mv2_crop_256/Spoof'
		self.live_img_root = '/home/kangcaixin/chenjiawei/siw-Mv2_crop_256/Live'
		self.protocol = pro
		self.spoof_type_list = list(self.spoof_type_dict.keys())
		if self.protocol == 3:
			self.protocol = 1

		if self.protocol == 1:
			self.unknown = 'None'
		elif self.protocol == 2:
			self.unknown = unknown
			assert self.unknown in self.spoof_type_list, print("Please offer a valid spoof type.")

		root_dir_id = "/home/kangcaixin/chenjiawei/ddpm-segmentation/face_anti_spoofing/dataset_SIWMv2/PROTOCOL/"
		self.spoof_train_fname = self.file_reader(root_dir_id + 'trainlist_all.txt')
		self.spoof_test_fname  = self.file_reader(root_dir_id + 'testlist_all.txt')
		self.live_train_fname  = self.file_reader(root_dir_id + 'trainlist_live.txt')
		self.live_test_fname   = self.file_reader(root_dir_id + 'testlist_live.txt')

	def file_reader(self, filename):
		with open(filename, 'r') as f:
			filenames_list = f.read().split('\n')
		return filenames_list

	# overriding the compile method.
	def compile(self, dataset_name='SiWM-v2'):
		'''generates train and test list for SIW-Mv2.'''
		self.LI_DATA_DIR = []
		self.SP_DATA_DIR = []
		self.LI_DATA_DIR_TEST = []
		self.SP_DATA_DIR_TEST = []
		# Train data.
		# a=0
		# b=0
		# c=0
		# d=0

		for x in self.live_train_fname:
			if x != '':
				self.LI_DATA_DIR.append(self.live_img_root + '/' + x)
				# if a==0:
				# 	print("live_train_fname:\n",self.live_img_root + '/Train/' +x)
				# 	a=a+1
		for x in self.spoof_train_fname:
			if x != '':
				if self.protocol == 1:
					self.SP_DATA_DIR.append(self.spoof_img_root + '/'+x[:x.rfind('_')]+'/'+ x)
					# if b==0:
					# 	print("spoof_train_fname\n:",self.spoof_img_root + '/'+x[:x.rfind('_')]+'/'+ x)
					# 	b=b+1
				elif self.protocol == 2:
					if self.spoof_type_dict[self.unknown] not in x:
						self.SP_DATA_DIR.append(self.spoof_img_root + '/'+x[:x.rfind('_')]+'/'+ x)
						# if b==0:
						# 	print("spoof_train_fname\n:",self.spoof_img_root + '/'+x[:x.rfind('_')]+'/'+ x)
						# 	b=b+1

		for x in self.live_test_fname:
			if x != '':
				self.LI_DATA_DIR_TEST.append(self.live_img_root + '/' + x)
				# if c==0:
				# 	print("live_test_fname:\n",self.live_img_root + '/Test/' + x)
				# 	c=c+1
		if self.protocol == 1:
			for x in self.spoof_test_fname:
				if x != '':
					self.SP_DATA_DIR_TEST.append(self.spoof_img_root + '/'+x[:x.rfind('_')]+'/'+ x)
					# if d==0:
					# 	print("spoof_test_fname:\n",self.spoof_img_root + '/'+x[:x.rfind('_')]+'/'+ x)
					# 	d=d+1
		elif self.protocol == 2:
			self.SP_DATA_DIR_TEST = glob(self.spoof_img_root +'/'+ self.spoof_type_dict[self.unknown] +'/' +self.spoof_type_dict[self.unknown] + '_*')
			# if d==0:
			# 	print("spoof_test_fname:\n", self.SP_DATA_DIR_TEST)
			# 	d=d+1

if __name__ == '__main__':
	config_siwm = Config_siwm(pro=2, unknown='Co')
	config_siwm.compile()