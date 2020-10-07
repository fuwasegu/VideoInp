# Originally proposed in [1]
# Implemented by cruller (http://yaritakunai.hatenablog.com/)
# [1] S. Gohshi and I. Echizen, "Limitations of Super Resolution Image
# Reconstruction and How to Overcome Them for a Single Image," Proc. SIGMAP
# 2013, Reykjavik, Iceland, pp. 71-78, July 2013.

import sys
from PIL import Image
import numpy as np
from scipy import ndimage

if len(sys.argv) != 3:
    print('Usage: python nlsp.py input output')
    sys.exit(1)

th = 121  # Threshold for Limiter [0..255]
alpha = 0.5  # Blend factor of the harmonics

img_in = Image.open(sys.argv[1])

# Upscaling
size = (img_in.width, img_in.height)
img_2x = img_in.resize(size, resample=Image.LANCZOS)
x = np.array(img_2x)

# High-Pass Filter
# Low-Pass Filter
lpf = np.array([[1, 2, 1],
                [2, 4, 2],
                [1, 2, 1]])/16  # Gaussian

# Convolve
x_lpf = np.empty(x.shape, dtype='uint8')
x_lpf[:, :, 0] = ndimage.convolve(x[:, :, 0], lpf)
x_lpf[:, :, 1] = ndimage.convolve(x[:, :, 1], lpf)
x_lpf[:, :, 2] = ndimage.convolve(x[:, :, 2], lpf)

# Subtractor: [-255, 255]
x_hpf = np.int16(x) - x_lpf
# /High-Pass Filter

# Non-Linear Function: [-65025, 65025]
x_nlf = np.sign(x_hpf)*np.int32(x_hpf)**2

# Limiter: [-255.0, 255.0]
c = (255 - th)/(65025 - th)
x_lmt = np.empty(x_nlf.shape)
# x_nlf < -th
x_lmt[x_nlf < -th] = c*(x_nlf[x_nlf < -th] + th) - th
# -th <= x_nlf <= th
ix = np.where((-th <= x_nlf) & (x_nlf <= th))
x_lmt[ix] = x_nlf[ix]
# th < x_nlf
x_lmt[th < x_nlf] = c*(x_nlf[th < x_nlf] - th) + th

# Adder: [-255, 510]
x_add = x + np.int16(np.around(alpha*x_lmt))

# Print clipped%
clip = x_add[x_add < 0].size + x_add[x_add > 255].size
print('Clipped: {:.2f}%'.format(clip/x_add.size*100))

# Clipping: [0, 255]
y = np.uint8(x_add)
y[x_add < 0] = 0
y[x_add > 255] = 255
img_out = Image.fromarray(y)

img_out.save(sys.argv[2])