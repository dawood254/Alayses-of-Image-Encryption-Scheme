import numpy as np
import matplotlib.pyplot as plt
import scipy as scipy
from scipy.stats import entropy
def image_histogram(gray_image):
  flat_image = gray_image.flatten()
  hist, bins = np.histogram(flat_image, bins=256, range=(0, 256))
  plt.hist(bins[:-1], bins, weights=hist, color='gray')
  plt.xlim([0, 256])
  plt.savefig("simple_plot.png")
  plt.show()

def Entropy(gray_image):
  flat_image = gray_image.flatten()
  hist, bins = np.histogram(flat_image, bins=256, range=(0, 256), density=True)
  Entropy = entropy(hist)
  print("Entropy:", Entropy)
def calculate_npc(original_image, watermarked_image):
    diff = original_image.astype(int) - watermarked_image.astype(int)
    n_pixels_changed = np.count_nonzero(diff)
    npc = n_pixels_changed / (original_image.shape[0] * original_image.shape[1])    
    return npc

def calculate_uaci(original_image, watermarked_image):
    diff = np.abs(original_image.astype(int) - watermarked_image.astype(int))
    sum_abs_diff = np.sum(diff)
    uaci = sum_abs_diff / (original_image.shape[0] * original_image.shape[1] * 255)
    return uaci

def calculate_psnr(original_image, encrypted_image):
    diff = original_image.astype(int) - encrypted_image.astype(int)
    mse = np.mean(diff**2)
    if mse == 0:
        return float("inf")
    else:
        return 20 * np.log10(255) - 10 * np.log10(mse)

def diagonal_correlation_analysis(image):
    x = image[:-1, :-1].flatten()
    y = image[1:, 1:].flatten()
    rand_index = np.random.permutation(x.size)
    rand_index = rand_index[:2000]
    x_rand = x[rand_index]
    y_rand = y[rand_index]

    plt.scatter(x_rand, y_rand, s=5, c='black', alpha=0.5)
    plt.show()

    diagnol_correlation = np.corrcoef(x_rand, y_rand)[0][1]
    return diagnol_correlation

def horizontal_correlation_analysis(Image):
    x = Image[:, :-1]
    y = Image[:, 1:]
    rand_index = np.random.permutation(x.size)
    rand_index = rand_index[:2000]
    x_rand = x.flatten()[rand_index]
    y_rand = y.flatten()[rand_index]

    plt.scatter(x_rand, y_rand, s=5, c='black', marker='o', alpha=0.5)
    plt.show()
    horizontal_correlation = np.corrcoef(x_rand, y_rand)[0, 1]
    
    return horizontal_correlation

def verticle_correlation_analysis(Image):
    x = Image[:-1,:]
    y = Image[1:,:]
    randIndex = np.random.permutation(x.size)
    randIndex = randIndex[:2000]
    xRand = x.ravel()[randIndex]
    yRand = y.ravel()[randIndex]
    plt.scatter(xRand, yRand, s=5, c='black', marker='o', alpha=1, linewidth=0)
    plt.show()
    correlation_verticle = np.corrcoef(xRand, yRand)[0,1]
    return correlation_verticle