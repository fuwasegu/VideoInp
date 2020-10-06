"""マスクを使って不要部分を補間するモジュール

"""
import numpy as np
import cv2
import os
from tqdm import tqdm
import math
import glob

global IMG_HEIGHT
global IMG_WIDTH
global BLACK
global WHITE

def getHomography(target_img_path, frame_img_path, mask_img_path):
    """ターゲットフレームと前後のフレームでホモグラフィ行列を計算し、射影変換する
    【引数】
        target_img_path: ターゲットフレームのパス。
        frame_img_path: 前後のフレームのパス。
        mask_img_path: マスク画像のパス。

    【戻り値】
        img: 射影変換後の画像のlist。
        　　　[射影変換後のフレーム画像, 射影変換後のマスク画像]

    """
    #変換される画像
    float_img = cv2.imread(frame_img_path, cv2.IMREAD_GRAYSCALE)
    #補間される画像
    ref_img = cv2.imread(target_img_path, cv2.IMREAD_GRAYSCALE)

    akaze = cv2.AKAZE_create()
    float_kp, float_des = akaze.detectAndCompute(float_img, None)
    ref_kp, ref_des = akaze.detectAndCompute(ref_img, None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(float_des, ref_des, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append([m])

    """
    img3 = cv2.drawMatchesKnn(float_img, float_kp, ref_img, ref_kp, good_matches, None, flags=2)
    # 画像表示
    cv2.namedWindow('img', cv2.WINDOW_NORMAL)
    cv2.imshow('img', img3)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    """

    #特徴点同士を結んだ直線をの傾きを計算して、射影変換を行うかを判断する
    #->傾きではなくて長さで分けたほうが良さそう
    low = []
    for i, lm in enumerate(good_matches):
        m = lm[0]
        p1 = tuple(map(int, float_kp[m.queryIdx].pt))
        p2 = tuple(map(int, ref_kp[m.trainIdx].pt))
        #xの増加量
        x_increace = int(p1[0]) - int(p2[0])
        #yの増加量
        y_increace = int(p1[1]) - int(p2[1])
        #length計算
        length = math.sqrt(x_increace*x_increace + y_increace*y_increace)
        #傾き計算
        try:
            a = x_increace / y_increace
        except ZeroDivisionError:
            a = 0

        if abs(a) < 10:
            low.append([m, length, a])


    #print(low)
    low = sorted(low, reverse=False, key=lambda x: x[1])
    del low[:int(len(low)*3/4)]
    #print(low)

    good_matches = []
    for i in low:
        a = []
        a.append(i[0])
        good_matches.append(a)

    """
    img3 = cv2.drawMatchesKnn(float_img, float_kp, ref_img, ref_kp, good_matches, None, flags=2)
    # 画像表示
    cv2.namedWindow('img', cv2.WINDOW_NORMAL)
    cv2.imshow('img', img3)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    """

    if len(good_matches) > 15:
        # 適切なキーポイントを選択
        ref_matched_kpts = np.float32(
            [float_kp[m[0].queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        sensed_matched_kpts = np.float32(
            [ref_kp[m[0].trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # ホモグラフィを計算
        H, status = cv2.findHomography(
            ref_matched_kpts, sensed_matched_kpts, cv2.RANSAC, 5.0)

        #カラー画像読み込み
        float_img2 = cv2.imread(frame_img_path)

        #マスク画像読み込み
        mask_img = cv2.imread(mask_img_path)

        # フレーム画像を変換
        warped_image = cv2.warpPerspective(
            float_img2, H, (float_img.shape[1], float_img.shape[0]))
        
        # マスク画像を変換
        warped_mask_image = cv2.warpPerspective(
            mask_img, H, (float_img.shape[1], float_img.shape[0]))
        
        #変換後の画像・変換後のマスク画像
        imgs = []
        imgs.append(warped_image)
        imgs.append(warped_mask_image)

        return imgs
    
    else:
        return False

def calc_interpolation_rate(mask):
    """画素の補間率を計算する
    【引数】
        mask: マスク画像

    【戻り値】
        whiteAreaRatio: 補間されていない部分の比率

    """
    this_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    image_size = this_mask.size
    whitePixels = cv2.countNonZero(this_mask)
 
    whiteAreaRatio = (whitePixels/image_size)*100#[%]

    print("White Area [%] : ", whiteAreaRatio)
    return whiteAreaRatio

def interpolation(target_img, frame_img, frame_mask_img, over_writtenable_mask_img,):
    """画素を補間する
    【引数】
        target_img: ターゲットフレームの画像
        frame_img: 射影変換後の前後フレーム画像
        frame_mask_img: 射影変換後のマスク画像
        over_writtenable_mask_img: 補間された画素を記録するためのマスク画像

    【戻り値】
        なし

    """
    for i in range(IMG_HEIGHT):
        for j in range(IMG_WIDTH):
            #マスク画像が白で、変換後画像が黒じゃない場合true
            if (over_writtenable_mask_img[i, j][0] == 255 and over_writtenable_mask_img[i, j][1] == 255 and over_writtenable_mask_img[i, j][2] == 255) and not(frame_img[i, j][0] == 0 and frame_img[i, j][1] == 0 and frame_img[i, j][2] == 0) and not(frame_mask_img[i, j][0] == 255 and frame_mask_img[i, j][1] == 255 and frame_mask_img[i, j][2] == 255):
                target_img[i, j] = frame_img[i, j]
                over_writtenable_mask_img[i, j] = BLACK

def get_homography_frames(target_img_path, target_index,in_frames_dir_path, frame_ext):
    """
    """
    homography_frames = []
    mask_img_path = '../mask.jpg'
    frame_name_list = sorted(glob.glob(in_frames_dir_path + '*.' + frame_ext))
    last_frame_index = 0
    zigzag_list = zigzag(target_index)
    
    for i in zigzag_list:
        if i == target_index:
            continue

        frame = getHomography(target_img_path, frame_name_list[i], mask_img_path)
        if frame == False:
            last_frame_index = i
            break
        else:
            homography_frames.append(frame)
            last_frame_index = i
    
    if i == 0:
        i = target_index*2 + 1
        while True:
            frame = getHomography(target_img_path, frame_name_list[i], mask_img_path)
            if frame == False:
                break
            else:
                homography_frames.append(frame)
                i = i + 1
    else:
        if i % 2 == 0:
            last_index = zigzag_list.index(i)
            start_index = target_index*2 - last_index + 1
            while start_index >= 0:
                frame = getHomography(target_img_path, frame_name_list[i], mask_img_path)
                if frame == False:
                    break
                else:
                    homography_frames.append(frame)
                    i = i - 1
        else:
            while start_index >= 0:
                frame = getHomography(target_img_path, frame_name_list[i], mask_img_path)
                if frame == False:
                    break
                else:
                    homography_frames.append(frame)
                    i = i - 1

    return homography_frames





    

def zigzag(center):
    """ジグザグにリストを返す。

    【備考】
	    インデックス０は中心の数字（自身）が入る。
	    偶数インデックスのベクトルは増加方向、奇数インデックスのベクトルは減少方向
    """
    result = []
    for i in range(center*2 + 1):
        if i % 2 == 0:
            i = i * (-1)
        center = center + i
        result.append(center)
    return result

def inpaint_frames(in_framse_dir_path, out_frame_dir_path, mask_path):
    pass




"""
if __name__ == "__main__":
    mask = cv2.imread('mask2.png')
    IMG_HEIGHT = mask.shape[0]
    IMG_WIDTH = mask.shape[1]
    BLACK = np.array([0, 0, 0])
    WHITE = np.array([255, 255, 255])
    path = '../frames/'
    files = os.listdir(path)
    fileCount = len(files)

    for target in tqdm(range(50, 51)):
        target_img_path = '../frames/img' + str(target) + '.jpg'
        target_img = cv2.imread(target_img_path)
        target_mask_img = cv2.imread('../../mask.jpg')

        for frame in tqdm(range(fileCount)):
            if frame == target:
                pass
            else:
                imgs = getHomography(target_img_path, '../frames/img' + str(frame) + '.jpg', '../../mask.jpg')
                interpolation(target_img, imgs[0], imgs[1], target_mask_img)
                rate = calc_interpolation_rate(target_mask_img)
                if rate < 1:
                    break
        
        cv2.imwrite('../../new_frames/' + str(target) + '.jpg', target_img)
        cv2.imwrite('../../new_masks/' + str(target) + '.jpg', target_mask_img)
"""