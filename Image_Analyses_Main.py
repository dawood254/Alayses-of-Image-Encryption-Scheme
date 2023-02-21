
import cv2
import Image_Analyses_fun
image = cv2.imread("image.jpg")
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
Image_Analyses_fun.image_histogram(gray_image)
Image_Analyses_fun.Entropy(gray_image)
Image_Analyses_fun.calculate_npc(gray_image,gray_image)
Image_Analyses_fun.calculate_uaci(gray_image,gray_image)
Image_Analyses_fun.calculate_psnr(gray_image,gray_image)
Image_Analyses_fun.diagonal_correlation_analysis(gray_image)
Image_Analyses_fun.horizontal_correlation_analysis(gray_image)
Image_Analyses_fun.verticle_correlation_analysis(gray_image)
