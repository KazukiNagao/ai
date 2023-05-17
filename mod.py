import cv2
import numpy as np
from PIL import Image

# MODNetをインポート
from MODNet.inference import MODNet
import torch

# MODNetモデルを初期化
modnet = MODNet(backbone_pretrained=False)
modnet.load_state_dict(torch.load("modnet_photographic_portrait_matting.ckpt", map_location=torch.device('cpu')))
modnet.eval()

# 背景を除去したい画像を読み込み
img = cv2.imread("dog.jpg")

# PIL Imageに変換
img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

# 背景除去を実行
_, _, alpha = modnet.predict(img)

# 背景を白くする
alpha = np.array(alpha * 255, dtype=np.uint8)
alpha = cv2.cvtColor(alpha, cv2.COLOR_GRAY2BGR)
white = np.ones_like(alpha, dtype=np.uint8) * 255
alpha = cv2.bitwise_not(alpha)
img = cv2.bitwise_and(img, alpha)
img = cv2.bitwise_or(img, white)

# 結果を保存
cv2.imwrite("output.jpg", img)
