

from IPython import get_ipython
get_ipython().magic('reset -sf')
import numpy as np
import cv2
from google.colab.patches import cv2_imshow
from matplotlib import pyplot as plt


input_image = cv2.imread('/content/Road_image2.jpg')

cv2_imshow(input_image)

######RGB to Gray Conversion
def rgb2gray(img):
  r = img[:,:,0]
  g = img[:,:,1]
  b = img[:,:,2]
  gray_image = 0.30 * r + 0.59 * g + 0.11 * b
  return  gray_image
gray = rgb2gray(input_image)   

cv2_imshow(gray)

###Sobel Filter
def sobel_filter(gray_input):
  Fx = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
  Fy = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
  [rows,coulmns] = gray_input.shape
  filtered = np.zeros_like(gray_input)

#filtered_image.shape

  for i in range(rows-2):
    for j in range(coulmns-2):
      gx = np.sum(np.multiply(Fx, gray_input[i:i + 3, j:j + 3])) 
      gy = np.sum(np.multiply(Fy, gray_input[i:i + 3, j:j + 3]))  
      filtered[i + 1, j + 1] = np.sqrt(gx ** 2 + gy ** 2)
  return filtered

filtered_image = sobel_filter(gray)
[rows,coulmns] = filtered_image.shape
for i in range(rows):
  for j in range(coulmns):
    if filtered_image[i,j]<200:
       filtered_image[i,j]=0
filtered_image = filtered_image.astype(int)
cv2_imshow(filtered_image)

#Hough Transform

def hough_transform(input):
  # selection of Rho and Theta ranges 
  Thetas = np.deg2rad(np.arange(-90.0, 90.0))   #Theta
  width, height = input.shape
  diag_len = round(np.sqrt(width**2 + height**2))   # max_dist for rho
  #diag_len = diag_len.astype(int)
  rhos = np.linspace(-diag_len, diag_len, diag_len * 2)

  # Cache some resuable values
  cos_t = np.cos(Thetas)
  sin_t = np.sin(Thetas)
  num_theta = len(Thetas)

  # theta vs rho accumulator array
  accumulator = np.zeros((2 * diag_len, num_theta), dtype=np.uint8)
  y_idxs, x_idxs = np.nonzero(input)  # (row, col) indexes to edges

  # Vote in the hough accumulator
  for i in range(len(x_idxs)):
    x = x_idxs[i]
    y = y_idxs[i]

    for t_idx in range(num_theta):
      # Calculate rho. diag_len is added for a positive index
      rho = round(x * cos_t[t_idx] + y * sin_t[t_idx])+diag_len
      accumulator[rho, t_idx] += 1

  return accumulator, Thetas, rhos 
acc, T, r = hough_transform(filtered_image)
#accumulator.shape
#cv2_imshow(accumulator)

plt.imshow(acc,cmap='gray',extent=[np.rad2deg(T[-1]),np.rad2deg(T[0]),r[-1],r[0]])
plt.title('Hough Transfor')
plt.xlabel('Angle in degrees',labelpad=1)
plt.ylabel('rho')
plt.axis('auto')
#for i in range(len(row)):
  #x,y = indices[i]
  #rho = r[x]
  #theta = np.rad2deg(T[y])
  #plt.plot(theta,rho,'o',20)

plt.show()

x = acc
m,n = x.shape
for i in range(m):
  for k in range(n):
    if x[i,k]<200:
      x[i,k]=0
row,coulmn = np.nonzero(x)
indices  = np.vstack((row,coulmn)).T

rhovec = []
thetavec = []
for i in range(len(indices)):        
  x,y = indices[i]
  rho = r[x]
  rhovec.append(rho)
  theta = T[y]
  thetavec.append(np.rad2deg(theta))
thetaF = []
rhoF = []
for i in range(len(thetavec)):
  if thetavec[i] not in thetaF:
    thetaF.append(thetavec[i])
    rhoF.append(rhovec[i])
####neglecting few lines based on angle theta
x = 0
for k in range(len(thetaF)):
  x += 1
  for i in range(x,len(thetaF)):
    if abs(thetaF[k]-thetaF[i])<6:
      thetaF[i]= 0
for i in range(len(rhoF)):
  if thetaF[i] == 0:
    rhoF[i] = 0
rhoF = np.array(rhoF)
rhoF = rhoF[rhoF!=0]
thetaF = np.array(thetaF)
thetaF = thetaF[thetaF!=0]
#L = np.empty([5, 1], dtype=int)    
for i in range(len(thetaF)):
  theta = thetaF[i]
  rho = rhoF[i]  
  a = np.cos(np.deg2rad(theta))
  b = np.sin(np.deg2rad(theta))
  x0 = a*rho
  y0 = b*rho
        # these are then scaled so that the lines go off the edges of the image
  x1 = int(x0 + 10000*(-b))
  y1 = int(y0 + 10000*(a))
  x2 = int(x0 - 10000*(-b))
  y2 = int(y0 - 10000*(a))
  

  cv2.line(input_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
cv2_imshow( input_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

#choose two angles corresponding to two lines given hieghest difference
diff=[]
theta=[]
x=0
for i in range(len(thetaF)):
  x+=1
  for j in range(x,len(thetaF)):
    theta.append((thetaF[i],thetaF[j]))
    diff.append(abs(thetaF[i]-thetaF[j]))
#print(diff)
#print(theta)
max_index = diff.index(max(diff))
#print('max_index =', max_index)
a = theta[max_index]
theta1 = a[0]
theta1_index=np.where(thetaF == theta1)
theta1 =np.deg2rad(theta1)
rho1 = rhoF[theta1_index]    
theta2=a[1]
theta2_index=np.where(thetaF == theta2)
theta2 =np.deg2rad(theta2)
rho2 = rhoF[theta2_index]
xintr = int((rho1*np.sin(theta2)-rho2*np.sin(theta1))/(np.cos(theta1)*np.sin(theta2)-np.sin(theta1)*np.cos(theta2)))
yintr = int((rho2*np.cos(theta1)-rho1*np.cos(theta2))/(np.cos(theta1)*np.sin(theta2)-np.sin(theta1)*np.cos(theta2)))
#print(xintr,yintr)
cv2.circle(input_image, (xintr,yintr), radius=10, color=(0, 0, 255), thickness=2)
cv2_imshow( input_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

from google.colab import drive
drive.mount('/content/drive')