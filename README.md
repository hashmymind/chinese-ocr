# chinese-ocr

## 正體中文OCR系統

	文字識別CTPN（tensorflow）
	文字辨識CRNN（pytorch）+CTC
	圖片生成Text Renderer

## 環境
	python 3.6.7 + tensorflow 1.14.0 + pytorch 1.4.1

## Demo
    python demo.py    
    
下載 [預訓練模型](#)

還沒Train好

### CRNN
將pytorch-crnn.pth放入/train/models中
### CTPN
將checkpoints.zip解壓後的內容放入/ctpn/checkpoints中

## 模型訓練

### CTPN訓練
詳見 [tensorflow-ctpn](https://github.com/eragonruan/text-detection-ctpn)

### CRNN訓練
#### 1.設定Text Renderer的字體、字庫、語料庫
修改`train/render/config`
在`train/render/data`下新增字體、字庫、語料庫
#### 2.訓練
	修改`train/config`
    執行`train/train.py`
#### 3.訓練結果
還沒Train好

## 效果展示
### CTPN
還沒Train好
### OCR
還沒Train好

## 參考
[warp-ctc-pytorch](https://github.com/SeanNaren/warp-ctc)
[chinese_ocr-(tensorflow+keras)](https://github.com/YCG09/chinese_ocr)
[CTPN-tensorflow](https://github.com/eragonruan/text-detection-ctpn)
[crnn-pytorch](https://github.com/meijieru/crnn.pytorch)