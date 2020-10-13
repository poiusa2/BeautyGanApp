#!/usr/bin/env python
# coding: utf-8

# In[3]:


import dlib
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.patches as patches
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
#conda install -c conda-forge dlib


# In[4]:


detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor('../models/shape_predictor_5_face_landmarks.dat') #landmark 찾아주는 함수


# In[5]:


img = dlib.load_rgb_image('../imgs/12.jpg')
plt.figure(figsize=(16,10))
plt.imshow(img)
plt.show()


# In[6]:


img_result = img.copy()
dets =detector(img, 1) #얼굴을 찾아준다
if len(dets) == 0: #사진에 얼굴 없을 경우
    print('cannot find faces!')
else:
    fig, ax = plt.subplots(1,figsize=(16,10))
    for det in dets: #dets는 영역들이 들어있음.
        x, y, w, h = det.left(), det.top(), det.width(), det.height()
        rect = patches.Rectangle((x,y),w,h,linewidth=2, edgecolor='r',facecolor='none')
        ax.add_patch(rect)
    ax.imshow(img_result)
    plt.show()


# In[7]:


fig, ax = plt.subplots(1, figsize=(16,10))
objs = dlib.full_object_detections() #얼굴이 삐뚤어졌을 때 돌려주는 코드
for detection in dets:
    s = sp(img, detection)
    objs.append(s)
    for point in s.parts():
        circle = patches.Circle((point.x, point.y),radius=3, edgecolor='r',facecolor='r')
        ax.add_patch(circle)
ax.imshow(img_result)


# In[8]:


faces = dlib.get_face_chips(img, objs, size=256, padding=0.3) #padding : 이미지 사이의 간격
fig, axes = plt.subplots(1, len(faces)+1, figsize=(20,16))
axes[0].imshow(img)
for i, face in enumerate(faces):
    axes[i+1].imshow(face)
    # 두번쨰 사진은 수평에 맞게 정렬
plt.show()

# In[9]:


def align_faces(img):
    dets = detector(img,1) #dets에 얼굴 영역정보
    objs = dlib.full_object_detections()
    for detection in dets: #얼굴들 중 얼굴 갯수만큼 . 
        s = sp(img,detection) #s : landmark (점에 대한 정보)
        objs.append(s) 
    faces = dlib.get_face_chips(img, objs, size= 256, padding=0.35) #얼굴 영역만 잘라 이미지를 만들어주는 것
    return faces 

test_img = dlib.load_rgb_image('../imgs/02.jpg')
test_faces = align_faces(test_img)
fig, axes = plt.subplots(1, len(test_faces)+1, figsize=(20,16))
axes[0].imshow(test_img)
for i, face in enumerate(test_faces):
    axes[i+1].imshow(face)
plt.show()

# # 화장 실행

# In[13]:


#모델 초기화
sess = tf.Session() 
sess.run(tf.global_variables_initializer())
saver = tf.train.import_meta_graph('../models/model.meta')
saver.restore(sess,tf.train.latest_checkpoint('../models'))
graph = tf.get_default_graph()
X = graph.get_tensor_by_name('X:0')
Y = graph.get_tensor_by_name('Y:0')
Xs = graph.get_tensor_by_name('generator/xs:0')


# In[14]:


def preprocess(img): #스케일링
    return (img /255. - 0.5) *2 
def deprocess(img): #원상복귀
    return (img+1)/2


# In[38]:


img1 = dlib.load_rgb_image('../imgs/no_makeup/vSYYZ639.png') #노메이크업
img1_faces = align_faces(img1)#수평정렬 및 
img2 = dlib.load_rgb_image('../imgs/makeup/vFG112.png')
img2_faces = align_faces(img2)  


# In[39]:


fig,axes= plt.subplots(1,2,figsize=(16,10))
axes[0].imshow(img1_faces[0])
axes[1].imshow(img2_faces[0])
plt.show()


# In[41]:


src_img = img1_faces[0]
ref_img = img2_faces[0]
X_img = preprocess(src_img)
X_img = np.expand_dims(X_img, axis=0) #차원 하나 늘려줌
Y_img = preprocess(ref_img)
Y_img = np.expand_dims(Y_img, axis=0)

output = sess.run(Xs, feed_dict={X:X_img, Y:Y_img}) #위에 작성한 X Y 값에 해당이미지를 준 것
output_img = deprocess(output[0]) #출력된 이미지를 rescaling

fig, axes = plt.subplots(1,3, figsize = (20,10))
axes[0].set_title('Source')
axes[0].imshow(src_img)
axes[1].set_title('Reference')
axes[1].imshow(ref_img)
axes[2].set_title('Result')
axes[2].imshow(output_img)
plt.show()

