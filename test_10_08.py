import cv2
import glob
import tqdm
from src import mask
from src import media
from src import inpainting

"""手順
１：mask.make_detail_mask()でディテールマスクを生成する
２：mask.make_range_mask()でレンジマスクを生成する
３：inpainting.get_homography_frames()で前後フレームを射影変換する
４：inpainting.inpaint_target()で不要部分補間をする
"""

media.movie2frames('../2020_10_08_test/test.mp4', '../2020_10_08_test/frames/', 'img')

mask.make_detail_mask('../2020_10_08_test/frames/', '../2020_10_08_test/d_mask.jpg', frame_num=300)

mask.make_range_mask('../2020_10_08_test/d_mask.jpg', '../2020_10_08_test/r_mask.jpg')

for index, img_path in enumerate(sorted(glob.glob('../2020_10_08_test/frames/*.jpg'))):
    if index < 50 or index > 100:
        continue
    result = inpainting.get_homography_frames(img_path, index, '../2020_10_08_test/frames/', '../2020_10_08_test/r_mask.jpg')
    target = cv2.imread(img_path)
    mask = cv2.imread('../2020_10_08_test/r_mask.jpg')
    img = inpainting.inpaint_target_ave_neer(target, result, mask)
    cv2.imwrite('../2020_10_08_test/result/img' + str(index) + '.jpg', img)

media.frames2movie('result.mp4', '../2020_10_08_test/result/', 'img', video_path='../2020_10_08_test/')