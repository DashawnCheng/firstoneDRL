import numpy as np 
import cv2 
import matplotlib.pyplot as plt
#读取两张图像
img1 = cv2.imread('1.png')
img2 = cv2.imread('2.png')

#测试图像是否被读取
#cv2.imshow('img1',img1)
#cv2.imshow('img2',img2)
#cv2.waitKey(0)
#寻找图片上的特征值，采用opencv自带的surf算法

orb = cv2.ORB_create(500)


kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)
 
#提取并计算特征点
bf = cv2.BFMatcher(cv2.NORM_HAMMING)
#knn筛选结果
#matches = bf.knnMatch(des1, trainDescriptors = des2, k = 2)
#good = [m for (m,n) in matches if m.distance < 0.75*n.distance]

#BF筛选结果
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)
print (kp1[1])
#查看最大匹配点数目
#print (len(good))
#img3 = cv2.drawKeypoints(img1,kp1,None,(255,0,0),4)
#plt.imshow(img3), plt.show()
#img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:80], img2, flags=2)
#plt.imshow(img3), plt.show()
print("数据类型:",type(kp1[0]))
print("关键点坐标:",kp1[0].pt)#第一个关键点位置坐标
print("邻域直径:",kp1[0].size)#关键点邻域直径

print("数据类型:",type(matches[1]))#查看类型
print("描述符之间的距离:",matches[1].distance)# 描述符之间的距离。越小越好。
print("图像中描述符的索引:",matches[1].queryIdx)#查询图像中描述符的索引
#求解相机位姿

#相机参数 fx,fy,cx,cy
fx = 525.0  # focal length x
fy = 525.0  # focal length y
cx = 319.5  # optical center x
cy = 239.5  # optical center y
#建立相机坐标系

#相机内参矩阵
cm = np.array([[fx,0,cx],
             [0,fy,cy],
             [0,0,1]],dtype=np.float64)
#相机畸变参数矩阵 因为不得知所以填None
dist = None

#世界坐标系中3D点的坐标(x,y,z)  选取最好的四个点进行pnp算法算位姿
objPoints = np.array([[kp1[0].pt[0],kp1[0].pt[1],0],
                      [kp1[1].pt[0],kp1[1].pt[1],0],
                      [kp1[2].pt[0],kp1[2].pt[1],0],
                      [kp1[3].pt[0],kp1[3].pt[1],0]],dtype=np.float64)

#图像坐标系中点的坐标，单位像素
imaPoints = np.array([kp2[0].pt,kp2[1].pt,
                     kp2[2].pt,kp2[3].pt],dtype=np.float64)

#旋转矩阵rvec，平移矩阵tvec 
retval,rvec,tvec = cv2.solvePnP(objPoints,imaPoints,cm,dist)
#将1*3的旋转矩阵转换为3*3的旋转矩阵
rotM = cv2.Rodrigues(rvec)[0]
#求解相机位姿
camera_postion = -np.matrix(rotM).T * np.matrix(tvec)
print(camera_postion.T)