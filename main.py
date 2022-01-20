import cv2
import math
import numpy as np
from nltk import download
from nltk.corpus import words
from paddleocr import PaddleOCR
from keras.models import load_model
from skimage.morphology import remove_small_objects

download('words')
word_dic = words.words() + ['pakistan','afghanistan','america','punjab','sindh','balochistan','kashmir']

MODEL = load_model('cnn_classifier.h5')

def convertToBinary(img_src):
    # img = cv2.imread(img_addr)
    ret,thresh = cv2.threshold(img_src,120,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    return thresh

def findDistance(d1,d2):
    distance = int(math.sqrt((d2[0]-d1[0])**2+(d2[1]-d1[1])**2))
    return distance

def cropImage(timg,tol=0):
    # Convert to binary
    # (thresh, timg) = cv2.threshold(timg, 127, 255, cv2.THRESH_BINARY)
    #  Crop only ones from binary
    mask = timg>tol
    timg = timg[np.ix_(mask.any(1),mask.any(0))]
    # Add padding to the image
    timg = np.pad(timg, pad_width=round(timg.shape[0]*0.3),mode='constant',constant_values = 0)
    timg = timg.astype(np.uint8)
    
    # Resize the image to 200*200 pixels
    # timg = cv2.resize(timg, (28,28), interpolation = cv2.INTER_AREA)
    timg = cv2.resize(timg, (28,28), interpolation = cv2.INTER_CUBIC)
    timg[timg > 0] = 255
    # timg = cv2.blur(timg,(25,25))
    return timg


def predictText(target_img,predicted):
    global MODEL,rgb_img
    target_img = convertToBinary(target_img)
    target_img = target_img.astype(bool)
    target_img = remove_small_objects(target_img,50)

    target_img = target_img.astype(np.uint8)
    n_labels,labels_im,stats,centroids = cv2.connectedComponentsWithStats(target_img)
    img_center_left = (0,int(labels_im.shape[0]/2))

    sep_imgs = []
    distances = []
    for lbl_i in range(1,n_labels):
        ttimg = np.zeros_like(labels_im)
        ttimg[labels_im == lbl_i] = 255
        rslt_img = cropImage(ttimg)
        distance = findDistance(img_center_left,centroids[lbl_i])
        distances.append(distance)
        sep_imgs.append((rslt_img,distance))
    distances = sorted(distances)

    sep_imgs_ordered = [False]*len(sep_imgs)
    for idx,sep_img in enumerate(sep_imgs):
        tidx = distances.index(sep_img[1])
        sep_imgs_ordered[tidx] = sep_img[0]


    sep_imgs_ordered = np.array(sep_imgs_ordered)
    predictions = np.argmax(MODEL.predict(sep_imgs_ordered),axis=1)
    CLASSES = ['FI','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
    text = []
    for pred in predictions:
        text.append(CLASSES[pred])
        text = ''.join(text)

    correct = text.lower() in word_dic
    color = (0,255,2) if correct else (255,0,0)
    coords = np.array(predicted[0])
    coords = coords.astype(int)
    thickness = 5
    rgb_img = cv2.line(rgb_img, tuple(coords[0]), tuple(coords[1]), color, thickness)
    rgb_img = cv2.line(rgb_img, tuple(coords[1]), tuple(coords[2]), color, thickness)
    rgb_img = cv2.line(rgb_img, tuple(coords[2]), tuple(coords[3]), color, thickness)
    rgb_img = cv2.line(rgb_img, tuple(coords[3]), tuple(coords[0]), color, thickness)
    rgb_img = cv2.putText(rgb_img, text, (coords[0][0]-50,coords[0][1]), cv2.FONT_HERSHEY_SIMPLEX, 3, color, thickness, cv2.LINE_AA)
    return correct



if __name__ == '__main__':
    ocr = PaddleOCR(use_angle_cls=True, lang='en') # need to run only once to download and load model into memory
    img_path = 'data/test1.jpg'
    result = ocr.ocr(img_path, cls=True)

    rgb_img = cv2.imread(img_path)
    img = cv2.cvtColor(rgb_img,cv2.COLOR_BGR2GRAY)

    word_imgs = []
    for line in result:
        location = np.array(line[0])
        timg = img[int(np.min(location[:,1])):int(np.max(location[:,1])),int(np.min(location[:,0])):int(np.max(location[:,0]))]
        word_imgs.append(timg)
        
    total = 0
    correct = 0
    for i,word_img in enumerate(word_imgs):
        total += 1
        if predictText(word_img,result[i]):
            correct += 1
    rgb_img = cv2.putText(rgb_img, f"(Correct/Total) = {correct}/{total}", (100,150), cv2.FONT_HERSHEY_SIMPLEX, 5, (0,0,255), 5, cv2.LINE_AA)
    cv2.imshow('Result Image',rgb_img)
