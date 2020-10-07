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

def get_homography_frames(target_img_path, target_index,in_frames_dir_path, mask_img_path, frame_ext='jpg'):
    """
    【返り値】
	    result: [変換後のフレーム[
                    [射影変換後のフレーム画像, 射影変換後のマスク画像], 
                        [射影変換後のフレーム画像, 射影変換後のマスク画像], ...
                    ], 
                    加重平均用のフレーム距離[int, int, int, ...]
                ]
    """
    homography_frames = []
    frame_name_list = sorted(glob.glob(in_frames_dir_path + '*.' + frame_ext))
    last_frame_index = 0
    zigzag_list = zigzag(target_index)

    #加重平均用のフレーム距離
    weights = []
    weight = 0
    counter = 0
    
    for i in zigzag_list:
        counter = counter + 1
        if i == target_index:
            continue

        frame = getHomography(target_img_path, frame_name_list[i], mask_img_path)
        if frame == False:
            last_frame_index = i
            break
        else:
            homography_frames.append(frame)
            last_frame_index = i
            if counter % 2 == 0:
                weight = weight + 1
            weights.append(weight)

    
    if i == 0:
        i = target_index*2 + 1
        while True:
            frame = getHomography(target_img_path, frame_name_list[i], mask_img_path)
            if frame == False:
                break
            else:
                homography_frames.append(frame)
                i = i + 1
                weights.append(weight)
            weight = weight + 1
    else:
        if zigzag_list.index(i)+1 % 2 == 0:
            last_index = zigzag_list.index(i)
            start_index = target_index*2 - last_index + 1
            while start_index >= 0:
                weight = weight + 1
                frame = getHomography(target_img_path, frame_name_list[i], mask_img_path)
                if frame == False:
                    break
                else:
                    homography_frames.append(frame)
                    i = i - 1
                    weights.append(weight)
        else:
            while i >= 0:
                frame = getHomography(target_img_path, frame_name_list[i], mask_img_path)
                if frame == False:
                    break
                else:
                    homography_frames.append(frame)
                    i = i - 1
                    weights.append(weight)
                weight = weight + 1

    max_weight = weights[-1] + 1
    weights_result = [max_weight - i for i in weights]
    result = []
    result.append(homography_frames)
    result.append(weights_result)
    return result

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

def inpaint_target_ave(target_img, in_frames, mask_img):
    """加重平均で補間
    【引数】
        target_img: ターゲットフレームの画像
        in_frames: 射影変換後のフレーム画像を格納したリスト
        mask_img: マスク画像のデータ

    【返り値】
        inpainted_target: インペインティング後のターゲット画像
    
    """
    heght = mask_img.shape[0]
    width = mask_img.shape[1]
    weight_list = in_frames[1]
    inpainted_target = target_img

    print('inpainting target frame...')
    for i in tqdm(range(heght)):
        for j in range(width):
            pixel_b = []
            pixel_g = []
            pixel_r = []
            weight = []
            for index, img in enumerate(in_frames[0]):
                frame = img[0]
                frame_mask = img[1]
                #マスクが白で、フレームが黒くない
                if mask_img[i, j][0] == 255 and mask_img[i, j][1] == 255 and mask_img[i, j][2] == 255:
                    if not(frame_mask[i, j][0] == 255 and frame_mask[i, j][1] == 255 and frame_mask[i, j][2] == 255):
                        if not(frame[i, j][0] == 0 and frame[i, j][1] == 0 and frame[i, j][2] == 0):
                            pixel_b.append(frame[i, j][0])
                            pixel_g.append(frame[i, j][1])
                            pixel_r.append(frame[i, j][2])
                            weight.append(weight_list[index])
                        else:
                            pass
                    else:
                        pass
                else:
                    pass
            if len(weight) > 0:
                inpainted_target[i, j][0] = np.average(pixel_b, weights=weight)
                inpainted_target[i, j][1] = np.average(pixel_g, weights=weight)
                inpainted_target[i, j][2] = np.average(pixel_r, weights=weight)
            else:
                pass

    print('done.')
    return inpainted_target

def inpaint_target_direct(target_img, in_frames, mask_img):
    """一番近い画素で補間
    【引数】
        target_img: ターゲットフレームの画像
        in_frames: 射影変換後のフレーム画像を格納したリスト
        mask_img: マスク画像のデータ

    【返り値】
        inpainted_target: インペインティング後のターゲット画像
    
    """
    heght = mask_img.shape[0]
    width = mask_img.shape[1]
    weight_list = in_frames[1]
    inpainted_target = target_img

    for index, img in enumerate(in_frames[0]):
        frame = img[0]
        frame_mask = img[1]
        mask_list = np.argwhere(frame_mask == 255)
        for i_j in mask_list:
            i = i_j[0]
            j = i_j[1]
            frame[i, j][0] = 0
            frame[i, j][1] = 0
            frame[i, j][2] = 0
        in_frames[0][index] = frame

    print('inpainting target frame...')
    white_point = np.argwhere(mask_img == 255)

    for index, i_j in enumerate(white_point):

        i = i_j[0]
        j = i_j[1]
        for img in in_frames[0]:
            if not(img[i, j][0] == 0 and img[i, j][1] == 0 and img[i, j][2] == 0):
                inpainted_target[i, j][0] = img[i, j][0]
                inpainted_target[i, j][1] = img[i, j][1]
                inpainted_target[i, j][2] = img[i, j][2]
                mask_img[i, j][0] = 0
                mask_img[i, j][1] = 0
                mask_img[i, j][2] = 0
                break

    print('done.')
    return inpainted_target

def inpaint_target_ave_neer(target_img, in_frames, mask_img):
    """加重平均値が最も近い実データ画素値で補間
    【引数】
        target_img: ターゲットフレームの画像
        in_frames: 射影変換後のフレーム画像を格納したリスト
        mask_img: マスク画像のデータ

    【返り値】
        inpainted_target: インペインティング後のターゲット画像
    
    """
    heght = mask_img.shape[0]
    width = mask_img.shape[1]
    weight_list = in_frames[1]
    inpainted_target = target_img

    print('inpainting target frame...')
    for i in tqdm(range(heght)):
        for j in range(width):
            pixel_b = []
            pixel_g = []
            pixel_r = []
            weight = []
            pixel_list_b = []
            pixel_list_g = []
            pixel_list_r = []
            for index, img in enumerate(in_frames[0]):
                frame = img[0]
                frame_mask = img[1]
                
                pixel_list_b.append(frame[i, j][0])
                
                pixel_list_g.append(frame[i, j][1])
                
                pixel_list_r.append(frame[i, j][2])

                #マスクが白で、フレームが黒くない
                if mask_img[i, j][0] == 255 and mask_img[i, j][1] == 255 and mask_img[i, j][2] == 255:
                    if not(frame_mask[i, j][0] == 255 and frame_mask[i, j][1] == 255 and frame_mask[i, j][2] == 255):
                        if not(frame[i, j][0] == 0 and frame[i, j][1] == 0 and frame[i, j][2] == 0):
                            pixel_b.append(frame[i, j][0])
                            pixel_g.append(frame[i, j][1])
                            pixel_r.append(frame[i, j][2])
                            weight.append(weight_list[index])
                        else:
                            pass
                    else:
                        pass
                else:
                    pass
            if len(weight) > 0:
                inpainted_target[i, j][0] = getNearestValue(pixel_list_b, np.average(pixel_b, weights=weight))
                inpainted_target[i, j][1] = getNearestValue(pixel_list_g, np.average(pixel_g, weights=weight))
                inpainted_target[i, j][2] = getNearestValue(pixel_list_r, np.average(pixel_r, weights=weight))
            else:
                pass

    print('done.')
    return inpainted_target

def getNearestValue(list, num):
    """
    概要: リストからある値に最も近い値を返却する関数
    @param list: データ配列
    @param num: 対象値
    @return 対象値に最も近い値
    """

    # リスト要素と対象値の差分を計算し最小値のインデックスを取得
    idx = np.abs(np.asarray(list) - num).argmin()
    return list[idx]
