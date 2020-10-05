"""マスクを使って不要部分を補間するモジュール

"""
import numpy as np
import cv2
import os
from tqdm import tqdm

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

def get_homography_frames(target_img, target_index,in_framse_dir_path):
    """
    """
    homography_frames = []


    pass

def zigzag(center):
    """
    """
    result = []
    for i in range(center*2):
        if i % 2 == 0:
            i = i * (-1)
        center = center + 1
        result.append(center)
    return result

def inpaint_frames(in_framse_dir_path, out_frame_dir_path, mask_path):
    pass





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