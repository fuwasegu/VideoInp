"""
knnによる特徴点マッチングのコード
"""

# -*- coding: utf-8 -*-
import cv2
import math

# 画像１
img1 = cv2.imread("img/img0.jpg")
# 画像２
img2 = cv2.imread("img/img99.jpg")
# 画像Aの横幅を調べておく
img1w = img1.shape[1]  


# A-KAZE検出器の生成
akaze = cv2.AKAZE_create()                                

# 特徴量の検出と特徴量ベクトルの計算
kp1, des1 = akaze.detectAndCompute(img1, None)
kp2, des2 = akaze.detectAndCompute(img2, None)

# Brute-Force Matcher生成
bf = cv2.BFMatcher()

# 特徴量ベクトル同士をBrute-Force＆KNNでマッチング
matches = bf.knnMatch(des1, des2, k=2)

# データを間引きする
ratio = 0.5
lowe = []
for m, n in matches:
    if m.distance < ratio * n.distance:
        lowe.append([m])

print(lowe)

# 対応する特徴点同士を描画
img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, lowe, None, flags=2)

# 対応する特徴点を取り出し上の画像の対応位置に
# 自前で円を描画してみる
print('特徴点の対応座標')
for i, lm in enumerate(lowe):
    # ペアごとに適当に色を生成
    bgr = tuple(i >> j & 1 for j in range(3))
    color1 = tuple(map(lambda x: x * 0xFF, bgr))
    color2 = tuple(map(lambda x: x * 0xC0, bgr))
    m = lm[0]
    p1 = tuple(map(int, kp1[m.queryIdx].pt))  # 画像Aのキーポイントから一致点を取り出す
    p2 = tuple(map(int, kp2[m.trainIdx].pt))  # 画像Bのキーポイントから一致点を取り出す
    cv2.circle(img3, p1, 3, color1, -1)
    cv2.circle(img3, (p2[0] + img1w, p2[1]), 3, color2, -1)
    print('(' + str(p1[0]).zfill(3) + ', ' + str(p1[1]).zfill(3) + ')', end='')
    print(' : ', end='')
    print('(' + str(p2[0]).zfill(3) + ', ' + str(p2[1]).zfill(3) + ')', end='')
    #xの増加量
    x_increace = int(p1[0]) - int(p2[0])
    #yの増加量
    y_increace = int(p1[1]) - int(p2[1])
    #傾き計算
    a = 0
    if x_increace != 0:
        a = y_increace / x_increace
    print('　傾き：' + '{:.05f}'.format(a), end='')
    #直線の長さ
    length = math.sqrt((x_increace * x_increace) + (y_increace * y_increace))
    print('　直線の長さ：' + '{:.05f}'.format(length))

#画像保存
cv2.imwrite('match.jpg', img3)

# 画像表示
cv2.imshow('img', img3)

# キー押下で終了
cv2.waitKey(0)
cv2.destroyAllWindows()
