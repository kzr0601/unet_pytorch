from unet import Unet
from getDataSets import MyDataset

import torch
from PIL import Image
import cv2
import os
from torchvision import transforms
import numpy as np
import time
import csv
from scipy import signal
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

resize_height = 540
resize_width = 640

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
plt.switch_backend("Agg")

def predict_img_seq(model, imgpath, respath):
    if not os.path.exists(respath):
        os.mkdir(respath)
    valid_transform = transforms.Compose([
        transforms.Resize([resize_height, resize_width]),
        transforms.ToTensor()
    ])

    files = os.listdir(imgpath)
    start = time.time()
    area_seq = []
    max_index, min_index, max_area, min_area = 0, 0, 0, resize_height*resize_width
    for file in files:
        filename = os.path.join(imgpath, file)
        img = cv2.imread(filename)
        if img is None:
            print(filename)
        index = file[:file.find(".")]
        height, width, channel = img.shape
        img_gray = Image.open(filename).convert('L')

        x_ = valid_transform(img_gray).reshape(1, 1, resize_height, resize_width)
        y = model(x_.to(device))
        img_y = torch.squeeze(y).cpu().detach().numpy()
        img_y[img_y>=0.5]=1
        img_y[img_y<0.5]=0
        img_y = img_y * 255
        img_y = cv2.resize(img_y.astype(np.uint8), (width, height))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        binary = cv2.morphologyEx(img_y, cv2.MORPH_OPEN, kernel)
        img_y = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        img[img_y != 0] = [255, 0, 0] #img[:, :, 0] = blue
        filename= os.path.join(respath, f"{index}.png")
        cv2.imwrite(filename, img_y)
        
        area = np.count_nonzero(img_y)
        area_seq.append(area)
        if area > max_area:
            max_area = area
            max_index = index
        if area < min_area:
            min_area = area
            min_index = index
    return max_index, max_area, min_index, min_area, area_seq

def tiff2imgseq(videopath, dirpath):
    if not os.path.exists(dirpath):
        os.mkdir(dirpath)
    #print("dirpath: ", dirpath)
    video = tc.opentiff(videopath)
    _, firstimg = video.retrieve()
    cv2.imwrite(os.path.join(dirpath, "0.png"), firstimg)
    index = 1
    for img in video:
        cv2.imwrite(os.path.join(dirpath, str(index)+".png"), img)
        index += 1
    video.release()

def little_fun(peaks1, peaks2, index, index0, index1):
    len1, len2 = len(peaks1[0]), len(peaks2[0])
    res = index
    if( index<len2 and (index1>=len1 or index1<len1 and peaks2[0][index] < peaks1[0][index1]) and (index0<0 or index0>=0 and peaks2[0][index] > peaks1[0][index0]) ):
        index += 1
        res = index
        while(index<len2 and (index1>=len1 or index1<len1 and peaks2[0][index] < peaks1[0][index1]) and (index0<0 or index0>=0 and peaks2[0][index] > peaks1[0][index0])):
            index += 1
    peaks2[0] = np.delete(peaks2[0], [range(res, index)])
    peaks2[1]['peak_heights'] = np.delete(peaks2[1]['peak_heights'], [range(res, index)])
    return res, peaks1, peaks2

def adjust_peaks(peaks, neg_peaks):
    index, peaks, neg_peaks = little_fun(peaks, neg_peaks, 0, -1, 0)
    for i in range(len(peaks[0])):
        index, peaks, neg_peaks = little_fun(peaks, neg_peaks, index, i, i+1)

    index2, neg_peaks, peaks = little_fun(neg_peaks, peaks, 0, -1, 0)
    for i in range(len(neg_peaks[0])):
        index2, neg_peaks, peaks = little_fun(neg_peaks, peaks, index2, i, i+1)

def get_heartrate(area_seq, imgname):
    tmp = area_seq
    mean_area = (np.mean(area_seq)+np.max(area_seq))/2
    peaks = list(signal.find_peaks(area_seq, height=mean_area, distance=20))  # distance=10
    neg_area_seq = np.max(area_seq)-area_seq
    mean_neg_area = (np.mean(neg_area_seq)+np.max(neg_area_seq))/2
    neg_peaks = list(signal.find_peaks(neg_area_seq, height=mean_neg_area, distance=20))
    adjust_peaks(peaks, neg_peaks)

    # draw heart area pic
    fig = plt.figure()
    x = [_ for _ in range(len(area_seq))]
    plt.plot(x, area_seq, 'b-')
    for i in range(len(peaks[0])):
        plt.scatter(x[peaks[0][i]],area_seq[peaks[0][i]], c='red', s=200, label='auto')
    for j in range(len(neg_peaks[0])):
        plt.scatter(x[neg_peaks[0][j]],area_seq[neg_peaks[0][j]], c='green', s=200, label='auto')
    plt.savefig(imgname)

    # cal heart rate
    res1  = 3000 / len(area_seq) * len(peaks)
    heartrate = np.zeros(len(peaks[0])-1)
    avg_heartrate = 0
    for i in range(1, len(peaks[0])):
        heartrate[i - 1] = (peaks[0][i] - peaks[0][i - 1])*0.02
        avg_heartrate += heartrate[i-1]
    return len(peaks), 60.0/(avg_heartrate/(len(heartrate)+1e-5)+1e-5), peaks, neg_peaks


def main():
    print("predict start now")
    modelpath = "./bestmodel_aug_100.pkl"
    data_path = "./video_data/2021010222"
    res_path = "./video_result"
    if not os.path.exists(res_path):
        os.mkdir(res_path)
 
    csv_name = os.path.join(res_path, "result.csv")
    csv_file = open(csv_name, 'w', encoding='utf-8')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["dir_name", "max_index", "max_area", "min_index", "min_area", "num_peaks", "heart_rate", "peak sequence", "neg peak sequence", "area sequence"]) #, "avg_heart_rate2", ])

    #model = Unet(1, 1)
    model = torch.load(modelpath)
    model = model.to(device)
 
    files = os.listdir(data_path)
    start = time.time()
    for file in files:
        print(file)
        filename = os.path.join(data_path, file)
        if(os.path.isdir(filename)):
            respath = os.path.join(res_path, file)
            max_index, max_area, min_index, min_area, area_seq = predict_img_seq(model, filename, respath)
            num_peaks, heart_rate, peak_seq, neg_peak_seq = get_heartrate(area_seq, os.path.join(res_path, f"{file}.png"))
            #avg_heart_rate1, avg_heart_rate2 = get_heartrate(area_seq, peaks)
            csv_writer.writerow([filename, max_index, max_area, min_index, min_area, num_peaks, heart_rate, peak_seq, neg_peak_seq, area_seq])
        elif(filename.split(".")[-1] == "tiff"):
            #print(file)
            dirpath = os.path.join(data_path, file.split(".")[0])
            if(os.path.exists(dirpath)):
                continue
            tiff2imgseq(filename, dirpath)
            
    csv_file.close()
	
    duration = time.time()-start
    print(f"predict {len(files)} videos take {duration}, avg video take {duration/len(files)}")
    print("end")

if __name__ == '__main__':
	main()
