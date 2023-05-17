from segment_anything import SamPredictor, sam_model_registry, SamAutomaticMaskGenerator
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from ultralytics import YOLO, yolo
# ultralytics.yolo.engine.results.Results


def image_trim():
    print("start trimming")
    model = YOLO('yolov8n-seg.pt')
    mask_img = cv2.imread("23VIT-S006-1-MlwnCRjSjdQkc2ip.jpg")
    mask_img = cv2.cvtColor(mask_img, cv2.COLOR_BGR2RGB)
    mask_img = cv2.cvtColor(mask_img, cv2.COLOR_RGB2GRAY)
    plt.imshow(mask_img, cmap='gray')
    plt.show()
    thresh, bin_img = cv2.threshold(mask_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    print("otsu: ", thresh)
    plt.imshow(bin_img, cmap='gray')
    plt.show()

    # 実行するとimgがエクスポートされる
    # results: List[yolo.engine.results.Results] = model.predict(source="23MLA-S522-1-HbK8Yhf33t7nkoJU.jpg", conf=0.27, save=True, boxes=False, show_labels=False, show_conf=False)
    # for r in results:
    #     print(r.masks.data)
    #     print(r.masks.xyn)
    original_img = cv2.imread("23VIT-S006/23VIT-S006-1-MlwnCRjSjdQkc2ip.jpg")

    white_img = np.zeros(mask_img.shape, dtype=np.uint8)
    mask = cv2.rectangle(white_img, (500, 500), (1800, 2000), (255, 255, 255), -1)
    cv2.imwrite("mask-1.jpg", mask)
    img_or = cv2.bitwise_and(mask_img, mask)
    plt.imshow(img_or)
    plt.show()
    # white_img.fill(255)
    # white_img = cv2.cvtColor(white_img, cv2.COLOR_BGR2GRAY)
    # white_img = cv2.cvtColor(white_img, cv2.COLOR_GRAY2BGR)
    return

    print(mask_img.shape)
    # セグメンテーションされたマスク画像を使って、背景を除去する
    # result_img = cv2.bitwise_and(white_img, mask_img)
    cv2.imwrite("masked.jpg", mask_img)
    # cv2.imwrite("result_before.jpg", result_img)
    _, mask = cv2.threshold(mask_img, 190, 255, cv2.THRESH_BINARY)
    cv2.imwrite("mask.jpg", mask)

    white = np.full(shape=original_img.shape, fill_value=(255, 255, 255), dtype='uint8')

    # 元画像と合成する
    # result_img = cv2.add(result_img, original_img)
    back = cv2.bitwise_and(white, white, mask=mask)
    cv2.imwrite("back.jpg", back)
    result_img = cv2.bitwise_or(original_img, back)

    # 結果を保存する
    cv2.imwrite("result.jpg", result_img)

    # sam_checkpoint = "./sam_vit_h_4b8939.pth"
    # model_type = "vit_h"
    # sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    # predictor = SamPredictor(sam)
    # predictor.set_image("./dog.jpg")
    # masks, _, _ = predictor.predict(<input_prompts>)
    # mask_generator = SamAutomaticMaskGenerator(sam)
    # image = cv2.imread('./dog.jpg')
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # plt.figure(figsize=(20, 20))
    # plt.imshow(image)
    # plt.axis('off')

    # masks = mask_generator.generate(image)
    # print(len(masks))
    # print(masks[0].keys())


image_trim()
