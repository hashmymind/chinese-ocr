class DefaultConfig(object):
	epoch_size = 1000000
	modelpath = 'D:/chinese-ocr/train/models/pytorch-crnn.pth'

	batch_size =4
	img_h = 32
	num_workers = 0
	use_gpu = True
	max_epoch = 10
	learning_rate = 2*(1e-4)
	weight_decay = 5*(1e-5)
	printinterval = 500
	valinterval = 6000

def parse(self,**kwargs):
	for k,v in kwargs.items():
		setattr(self,k,v)

DefaultConfig.parse = parse
opt = DefaultConfig()