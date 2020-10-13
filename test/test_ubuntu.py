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

#mask.make_detail_mask('../frames/', '../test/d_mask.jpg', frame_num=300)
#mask.make_range_mask('../test/d_mask.jpg', '../test/r_mask.jpg')

for index, img_path in enumerate(sorted(glob.glob('../frames_test/*.jpg'))):
    if index < 100:
        continue
    result = inpainting.get_homography_frames(img_path, index, '../frames_test/', '../test/r_mask.jpg')
    target = cv2.imread(img_path)
    mask = cv2.imread('../test/r_mask.jpg')
    img = inpainting.inpaint_target_ave_neer(target, result, mask)
    cv2.imwrite('../test/img_' + str(index) + '.jpg', img)

