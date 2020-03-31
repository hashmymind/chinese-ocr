import sys
import os
# os.environ['CUDA_VISIBLE_DEVICES']='0'
import torch
from config import opt
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms as T
import cv2
from PIL import Image
import numpy as np
from crnn import crnn
from torch.nn import CTCLoss
import torch.optim as optim
from torchvision import transforms
import collections
from tensorboardX import SummaryWriter
import gc
from render.main import gener

writer=SummaryWriter()
img_h=opt.img_h
batch_size=opt.batch_size
use_gpu=opt.use_gpu
max_epoch=opt.max_epoch

char_set = open('char_std.txt', 'r', encoding='utf-8').readlines()
char_set = ''.join([ch.strip('\n') for ch in char_set[1:]] + ['卍'])
gener = gener()
def readfile(filename):
	res = []
	with open(filename, 'r',encoding='utf-8') as f:
		lines = f.readlines()
		for i in lines:
			res.append(i.strip())
	dic = {}
	for i in res:
		p = i.split('卍')
		dic[p[0]] = p[1:]
	return dic

class resizeNormalize(object):
	def __init__(self, size, interpolation=Image.BILINEAR):
		self.size = size
		self.interpolation = interpolation
		self.toTensor = transforms.ToTensor()

	def __call__(self, img):
		img = img.resize(self.size, self.interpolation)
		img = self.toTensor(img)
		img.sub_(0.5).div_(0.5)
		return img



class Chineseocr(Dataset):

	def __init__(self):
		pass
		#self.image_dict = readfile(labelroot)
		#self.image_root = imageroot
		#self.image_name = [filename for filename, _ in self.image_dict.items()]

	def __getitem__(self, index):
		length = 0
		'''
		while length < 10:
			img_path = os.path.join(self.image_root, self.image_name[index])
			keys = self.image_dict.get(self.image_name[index])
			label = [char_set.index(x) for x in list(keys[0])]
			length = len(label)
			index+=1
		'''
		'''
		
		img_path = os.path.join(self.image_root, self.image_name[index])
		keys = self.image_dict.get(self.image_name[index])
		label = [char_set.index(x) for x in list(keys[0])]
		'''
		im, word = next(gener)
		label = [char_set.index(x) for x in list(word)]
		
		if len(im.shape) == 3:
			im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
		im = np.expand_dims(im, axis=0)
		im = torch.from_numpy(im)
		im = im.type(torch.FloatTensor)
		im.sub_(0.5).div_(0.5)
		#print(im.shape)
		'''
		Data = Image.open(img_path).convert('L')

		(w,h) = Data.size
		size_h = 32
		ratio = 32 / float(h)
		size_w = int(w * ratio)
		transform = resizeNormalize((size_w,size_h))
		Data = transform(Data)
		print(Data.shape)
		'''
		label=torch.IntTensor(label)
		return im,label

	def __len__(self):
		return opt.epoch_size



train_data = Chineseocr()
train_loader = DataLoader(
	train_data,
	batch_size = opt.batch_size,
	shuffle = True,
	num_workers = opt.num_workers
)

val_data = Chineseocr()
val_loader = DataLoader(
	val_data,
	batch_size = 1,
	shuffle = True,
	num_workers = opt.num_workers
)

def decode(preds):
	pred = []
	s = ""
	#print(preds)
	for i in range(len(preds)):
		if preds[i] != 0 and ((i == 0) or (i != 0 and preds[i] != preds[i-1])):
			pred.append(int(preds[i]))
			s += char_set[preds[i]]
	# showing
	print(s)
	return pred

def val(net,loss_func,max_iter = 100):
	print('start val')

	net.eval()
	totalloss = 0.0
	k = 0
	correct_num = 0
	total_num = 0
	val_iter = iter(val_loader)
	max_iter = min(max_iter,len(val_loader))
	for i in range(max_iter):
		if i % 50 == 0:
			gc.collect()
		k = k + 1
		(data,label) = val_iter.next()
		data = data.cuda()
		label = label.cuda()
		labels = torch.IntTensor([]).cuda()
		for j in range(label.size(0)):
			labels = torch.cat((labels,label[j]),0)

		output = net(data)
		output_size = torch.IntTensor([output.size(0)] * int(output.size(1)))
		label_size = torch.IntTensor([label.size(1)] * int(label.size(0)))
		loss = loss_func(output, labels, output_size, label_size) / label.size(0)
		totalloss += float(loss)
		pred_label = output.max(2)[1]
		pred_label = pred_label.transpose(1,0).contiguous().view(-1)
		pred = decode(pred_label)
		total_num += len(pred)
		for x,y in zip(pred,labels):
			if int(x) == int(y):
				correct_num += 1
	accuracy = correct_num / float(total_num) * 100
	test_loss = totalloss / k

	print('Test loss : %.3f Accuracy: %.3f' % (test_loss,accuracy))




if __name__ == '__main__':
	
	n_class = len(char_set)

	model = crnn.CRNN(img_h, 1, n_class, 256).cuda()

	modelpath = opt.modelpath

	learning_rate = opt.learning_rate
	if opt.batch_size ==1:
		loss_func = CTCLoss(blank=0, reduction='none').cuda()
	else:
		loss_func = CTCLoss(blank=0, reduction='sum').cuda()
	optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=opt.weight_decay)

	if os.path.exists(modelpath):
		print('Load model from "%s" ...' % modelpath)
		model.load_state_dict(torch.load(modelpath))
		print('Done!')
	k = 0
	losstotal = 0.0
	printinterval = opt.printinterval
	valinterval = opt.valinterval
	numinprint = 0
	# train

	for epoch in range(max_epoch):
		for i,(data,label) in enumerate(train_loader):
			data = data.cuda()
			label = label.cuda()
			k = k + 1
			numinprint = numinprint + 1
			model.train()
			labels = torch.IntTensor([]).cuda()
			for j in range(label.size(0)):
				labels = torch.cat((labels,label[j]),0)
			output = model(data)
			output_size = torch.IntTensor([output.size(0)] * int(output.size(1))).cuda()
			label_size = torch.IntTensor([label.size(1)] * int(label.size(0))).cuda()

			loss = loss_func(output,labels,output_size,label_size) / label.size(0)
			losstotal += float(loss)
			if k % printinterval == 0:
				print("[%d/%d] || [%d/%d] || Loss:%.3f" % (epoch,max_epoch,i + 1,len(train_loader),losstotal / numinprint))
			if k % 500 == 0:
				print("Saving ...")	
				torch.save(model.state_dict(), opt.modelpath+str(int((k/500)%10)))
				print("Done")
			losstotal = 0.0
			numinprint = 0
			writer.add_scalar('loss', loss, k)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			if k % valinterval == 0:
				# val
				val(model,loss_func)
			if k % 1000 == 0:
				gc.collect()
		# print('epoch : %05d || loss : %.3f' % (epoch, losstotal/numinepoch))



	writer.export_scalars_to_json("./all_scalars.json")
	writer.close()

