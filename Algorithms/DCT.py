import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.cm as cm

B = 8  # blocksize
fn3 = '../../Photos/Radar.png'
img1 = cv2.imread(fn3, cv2.IMREAD_GRAYSCALE)
h, w = np.array(img1.shape[:2]) // B * B
print(h)
print(w)
img1 = img1[:h, :w]

blocksV = h // B
blocksH = w // B
vis0 = np.zeros((h, w), np.float32)
vis0[:h, :w] = img1

# DCT transformation on the entire image
Trans = cv2.dct(vis0)

cv2.imwrite('Transformed.jpg', Trans)

row, col = 0, 0  # Set to the top-left corner or any other point you prefer

plt.imshow(img1, cmap="gray")
plt.plot([B * col, B * col + B, B * col + B, B * col, B * col], [B * row, B * row, B * row + B, B * row + B, B * row])
plt.axis([0, w, h, 0])
plt.title("Original Image")

plt.figure()
plt.subplot(1, 2, 1)
selectedImg = img1[row * B:(row + 1) * B, col * B:(col + 1) * B]
N255 = Normalize(0, 255)
plt.imshow(selectedImg, cmap="gray", norm=N255, interpolation='nearest')
plt.title("Image in selected Region")

plt.subplot(1, 2, 2)
selectedTrans = Trans[row * B:(row + 1) * B, col * B:(col + 1) * B]
plt.imshow(selectedTrans, cmap=cm.jet, interpolation='nearest')
plt.colorbar(shrink=0.5)
plt.title("DCT transform of selected Region")

plt.show()
