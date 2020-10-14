import sys
import os
import cv2
import glob
import tqdm

sys.path.append('../')
from src import mask
from src import media
from src import inpainting

"""手順
１：mask.make_detail_mask()でディテールマスクを生成する
２：mask.make_range_mask()でレンジマスクを生成する
３：inpainting.get_homography_frames()で前後フレームを射影変換する
４：inpainting.inpaint_target()で不要部分補間をする
"""

#dir_path =  sys.argv[1] #varificatioin number
base_path = '../../2020_10_13_2_test/'
movie_path = '../../2020_10_13_2_test/test.mp4'

media.movie2frames(movie_path, base_path + 'frames/', 'img')

mask.make_detail_mask(base_path + 'frames/', base_path + 'd_mask.jpg', frame_num=300)

mask.make_range_mask(base_path + 'd_mask.jpg', base_path + 'r_mask.jpg')

os.makedirs(base_path + 'result', exist_ok=True)
for index, img_path in enumerate(sorted(glob.glob(base_path + 'frames/' + '*.jpg'))):
    if index < 50 or index > 140:
        continue
    result = inpainting.get_homography_frames(img_path, index, base_path + 'frames/', base_path + 'r_mask.jpg')
    target = cv2.imread(img_path)
    mask = cv2.imread(base_path + 'r_mask.jpg')
    img = inpainting.inpaint_target_ave(target, result, mask)

    cv2.imwrite(base_path + 'result/img' + str(index) + '.jpg', img)

media.frames2movie('result.mp4', base_path + 'result/', 'img', video_path=base_path)