"""マスク画像生成に関するモジュール

"""
import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import glob
import pyclustering
from pyclustering.cluster import xmeans

def create_blank_mask(height, width):
    """マスク用のブランク画像（真っ黒の画像）を生成する。
    【引数】
        height: ブランク画像の高さ
        width: ブランク画像の幅
    """
    blank = np.zeros((height, width, 1))
    return blank


def make_detail_mask(dir_path, mask_path, frame_ext='jpg', frame_num=100, threshold=70, show_flag=0):
    """分散を計算してディティールマスクを生成する
    【引数】
        dir_path: 使用するフレームのあるディレクトリのパス。
        mask_path: 出力するディティールマスク画像のパス。
        frame_ext: 使用するフレームの拡張子。デフォルトはjpg。
        frame_num: 分散の計算に使用するフレームの枚数。デフォルトは100枚。
        threshold: マスクする不要部分の分散の閾値。デフォルトは70。
        show_flag: マスク画像生成後、画像を表示するかどうか。０だと表示しない。１だと表示する。

    【戻り値】
        なし。

    """
    frames = []
    frame_counter = frame_num
    print('Loading frames...')
    for frame in tqdm(sorted(glob.glob(dir_path + '*.' + frame_ext))):
        frames.append(cv2.imread(frame, cv2.IMREAD_GRAYSCALE))
        frame_counter -= 1
        if frame_counter < 0:
            break
    
    height = frames[0].shape[0]
    width = frames[0].shape[1]

    blank = create_blank_mask(height, width)

    print('Calculating the variance of the frame...')
    for i in tqdm(range(height)):
        for j in range(width):
            pixels = []
            for k in range(frame_num):
                pixels.append(frames[k][i, j])
            a = np.var(pixels)
            
            if (a < threshold):
                blank[i, j] = 255
            else:
                blank[i, j] = 0
    
    mask = blank

    cv2.imwrite(mask_path, mask)

    if show_flag == 0:
        return
    elif show_flag == 1:
        cv2.imshow('window', mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return
    else:
        return
    

def make_range_mask(in_mask_path, out_mask_path, show_flag=0):
    """x-meansを使ってレンジマスクを生成する
    【引数】
        in_mask_path: 入力するディティールマスク画像のパス
        out_mask_path: 出力するレンジマスク画像のパス
        show_flag: マスク画像生成後、画像を表示するかどうか。０だと表示しない。１だと表示する。

    【返り値】
        なし

    """
    in_mask = cv2.imread(in_mask_path, cv2.IMREAD_GRAYSCALE)

    height = in_mask.shape[0]
    width = in_mask.shape[1]

    X = []

    print('Checking unnecessary part...')
    for i in tqdm(range(height)):
        for j in range(width):
            if (in_mask[i, j] == 255):
                X.append([i, j])

    initializer = xmeans.kmeans_plusplus_initializer(data=X, amount_centers=2)
    initial_centers = initializer.initialize()
    xm = xmeans.xmeans(data=X, initial_centers=initial_centers)
    xm.process()
    clusters = xm.get_clusters()

    mask = create_blank_mask(height, width)
    print('Clustering unnecessary part...')
    for cluster in tqdm(clusters):
        coodinates = []
        for item in cluster:
            coodinates.append(X[item])

        x, y, width, height = cv2.boundingRect(np.array(coodinates))

        for i in range(y, y + height):
            for j in range(x, x + width):
                mask[j, i] = 255

    cv2.imwrite(out_mask_path, mask)
    if show_flag == 0:
        return
    elif show_flag == 1:
        cv2.imshow('window', mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return
    else:
        return


def plot_unnecessary_part_by_clustering(in_mask_path, in_img_path, out_img_path, show_flag=0):
    """ディティールマスク部を画像に矩形として表示する
    【引数】
        in_mask_path: 入力するディティールマスクのパス
        in_img_path: 矩形を重ねたいフレーム画像のパス
        out_img_path: 出力する矩形描画後の画像のパス
        show_flag: マスク画像生成後、画像を表示するかどうか。０だと表示しない。１だと表示する。

    【返り値】
        なし

    """
    in_mask = cv2.imread(in_mask_path, cv2.IMREAD_GRAYSCALE)

    height = in_mask.shape[0]
    width = in_mask.shape[1]

    X = []

    print('Checking unnecessary part...')
    for i in tqdm(range(height)):
        for j in range(width):
            if (in_mask[i, j] == 255):
                X.append([i, j])

    initializer = xmeans.kmeans_plusplus_initializer(data=X, amount_centers=2)
    initial_centers = initializer.initialize()
    xm = xmeans.xmeans(data=X, initial_centers=initial_centers)
    xm.process()
    clusters = xm.get_clusters()

    img_out = cv2.imread(in_img_path)
    mask = create_blank_mask(height, width)
    print('Clustering unnecessary part...')
    for cluster in tqdm(clusters):
        coodinates = []
        for item in cluster:
            coodinates.append(X[item])

        x, y, width, height = cv2.boundingRect(np.array(coodinates))
        img_out = cv2.rectangle(img_out, (y, x), (y + height, x + width), (0, 0, 255), 2)
        for i in range(y, y + height):
            for j in range(x, x + width):
                mask[j, i] = 255

    cv2.imwrite(out_img_path, img_out)
    if show_flag == 0:
        return
    elif show_flag == 1:
        cv2.imshow('window', img_out)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return
    else:
        return

