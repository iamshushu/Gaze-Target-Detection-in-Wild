train_annotation_dir = '/home/soo/gaze_total/gaze/DAM/data/videoattentiontarget/annotations/train/All in the Family/1023_1249/s00.txt'
data_dir = '/home/soo/gaze_total/gaze/DAM/data/videoattentiontarget/images/All in the Family/1023_1249'

import cv2
img = cv2.imread(data_dir + '/00001029.jpg')
img = cv2.rectangle(img, (181,46), (439, 349), (255, 255, 0), 3)
img = cv2.rectangle(img, (632,90), (955, 449), (255, 0, 0), 3)
img = cv2.circle(img, (212,449), 5, (255,255,0),3)
cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()



