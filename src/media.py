"""動画、画像の分割、統合系のモジュール
"""

import os
import cv2
import glob


def movie2frames(video_path, dir_path, basename, ext='jpg'):
    """入力した動画をフレームに切って指定したディレクトリに保存する。
    【引数】
        video_path: 入力する動画のパス
        dir_path: フレームを出力するディレクトリのパス
        basename: 保存するフレームのファイル名の共通部分
        ext: 保存するフレームの拡張子。デフォルトはjpg。

    【返り値】
        なし

    【使用例】
        >>> movie2frames('movie.mp4', 'frames/', 'frame', 'png')
        frames/frame1.png, frames/frame2.png, frames/frame3.png, frames/frame4.png, ...

    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return

    os.makedirs(dir_path, exist_ok=True)
    base_path = os.path.join(dir_path, basename)
    digit = len(str(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))))
    num = 1

    while True:
        ret, frame = cap.read()
        if ret:
            cv2.imwrite('{}{}.{}'.format(base_path, str(num).zfill(digit), ext), frame)
            num += 1
        else:
            return

def frames2movie(video_name, dir_path, basename, fps=30, frame_ext='jpg', video_path='./'):
    """ディレクトリ内の全てのフレームを使って動画を生成する。
    【引数】
        video_name: 生成する動画のファイル名。拡張子付き。
        dir_path: 動画生成に使用するフレームのディレクトリパス
        basename: 動画生成に使用するフレームのファイル名の共通部分
        fps: 生成される動画のフレームレート。デフォルトは30。
        frame_ext: 動画生成に使用するフレームの拡張子。デフォルトはjpg。
        video_path: 動画を保存するディレクトリのパス。デフォルトはカレントディレクトリ。

    【返り値】
        なし

    【使用例】
        >>> frame2movies('movie.mp4', 'frmames/', 'frame', 'png')
        ./movie.mp4
    """
    img_array = []

    for filename in sorted(glob.glob(dir_path + basename + '*.' + frame_ext)):
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)

    out = cv2.VideoWriter(video_path+video_name, 0x7634706d, fps, size)

    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()

def read_frames(frames_dir_path, frame_ext):
    """フレームを全て読み込む
    【引数】
        frames_dir_path: フレームがあるディレクトリのパス
        frame_ext: フレーム画像の拡張子

    【戻り値】
        frames: フレーム画像が格納されたリスト
            frames = [img1, img2, img3, ...]
    
    """

    frames = []

    for filename in sorted(glob.glob(frames_dir_path + '*.' + frame_ext)):
        frame = cv2.imread(filename)
        frames.append(frame)
    
    return frames