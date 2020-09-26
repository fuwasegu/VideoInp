"""マスク画像生成に関するモジュール

"""
import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import glob

def create_blank_mask(height, width):
    """マスク用のブランク画像（真っ黒の画像）を生成する。
    【引数】
        height: ブランク画像の高さ
        width: ブランク画像の幅
    """
    blank = np.zeros((height, width, 1))
    return blank

def make_mask_from_variance(dir_path, mask_path_and_name, frame_ext='jpg', frame_num=100, threshold=70, show_flag=0):
    """N枚のフレームの画素値の分散から、不要部分を検知する。
    【引数】
        dir_path: 使用するフレームのあるディレクトリのパス。
        mask_path_and_name: 出力するマスク画像のパスとファイル名。
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
        if frame_counter <= 0:
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

    cv2.imwrite(mask_path_and_name, mask)

    if show_flag == 0:
        return
    elif show_flag == 1:
        cv2.imshow('window', mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return
    else:
        return
    