import numpy as np 

import numpy as np
import cv2
from dataset_utils import load_dataset
 
def extract_eyes(img):
  #face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
  eye_cascade = cv2.CascadeClassifier('../cascades/haarcascade_eye.xml')

  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  eyes = eye_cascade.detectMultiScale(gray)
  return eyes
 
import random
# doesn't use np.shuffle.random, because of the chance of 2 images coinciding, so instead we rotate the array
def shuffleArr(images):
  shiftAmount = random.randint(1, images.shape[0] - 1)
  shuffled = np.copy(images)
  for idx, image in enumerate(images):
    new_location = (idx + (len(images) - shiftAmount)) % len(images)
    shuffled[new_location] = image
  return shuffled
 
def exchange_eye(im_f, im_s, eye_f, eye_s):
  (x_coord_first, y_coord_first, deltaX_first, deltaY_first) = eye_f
  (x_coord_second, y_coord_second, deltaX_second, deltaY_second) = eye_s
  cropped_eye = np.empty([deltaX_second + 1, deltaY_second + 1, 3], dtype=np.uint8)
 
  for row in range(y_coord_second, y_coord_second + deltaY_second + 1):
      for col in range(x_coord_second, x_coord_second + deltaX_second + 1):

       row = min(127, row)
       col = min(127,col)
       cropped_eye[row - y_coord_second][col - x_coord_second] = im_s[row][col]
 
  cropped_eye_resized = resizeImg(cropped_eye, deltaY_first + 1, deltaX_first + 1)
 
  for row in range(y_coord_first, y_coord_first + deltaY_first + 1):
      for col in range(x_coord_first, x_coord_first + deltaX_first + 1):
       
       row = min(127, row)
       col = min(127,col)
       im_f[row][col] = cropped_eye_resized[row - y_coord_first][col - x_coord_first] 
 
  # first_eye_pos = im_s[y_coord_first: y_coord_first + deltaY_first + 1][x_coord_first: x_coord_first + deltaX_first + 1]
  # second_eye_pos = im_s[y_coord_second: y_coord_second + deltaY_second + 1][x_coord_second: x_coord_second + deltaX_second + 1]
  # second_eyed_resized = resizeImg(second_eye_pos, deltaY_first, deltaX_first)
 
  # first_eye_pos = second_eye_pos
 
from PIL import Image
 
def save_image(first_image, saving_dir, fileName):
  image = Image.fromarray(first_image).convert('RGB')
  image.save(saving_dir + '/neg_' + str(fileName) + ".png")
 
#### Interpolations:
# INTER_CUBIC 
# INTER_AREA 
# INTER_LINEAR
# INTER_NEAREST  
# INTER_LANCZOS4
def resizeImg(image, width, height, interpolation = cv2.INTER_CUBIC):
  return cv2.resize(image, dsize=(width, height), interpolation = interpolation)
 
def generate_dataset(path, saving_dir):
  images = load_dataset(path).astype(np.uint8)[2494:]
  fileName = 2494
  for (first_image, second_image) in zip(images, shuffleArr(images)):
    
    print("next image")
    first_eyes, second_eyes = extract_eyes(first_image), extract_eyes(second_image)
    if len(first_eyes) != 2 or len(second_eyes) != 2:
      continue

    for (eye_first_person, eye_second_person) in zip(first_eyes, second_eyes):
      exchange_eye(first_image, second_image, eye_first_person, eye_second_person)
    save_image(first_image, saving_dir, fileName)
    fileName += 1

if __name__ == "__main__":

    generate_dataset('../data/test_np', '../data/test_cut')