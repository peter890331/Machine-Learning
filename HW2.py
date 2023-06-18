import numpy as np
import numpy.random as rd
import matplotlib
matplotlib.use('TkAgg')         #pycharm bug
import matplotlib.pyplot as plt
#import matplotlib.image as img
import cv2
from sklearn.decomposition import PCA
import glob
#from mlxtend.plotting import plot_decision_regions
#from sklearn.svm import SVC
from tqdm import tqdm
import time

images_train_Carambula = []
images_train_Lychee = []
images_train_Pear = []

path_train_Carambula = glob.glob("Data_train\\Carambula\\*.png")
path_train_Lychee = glob.glob("Data_train\\Lychee\\*.png")
path_train_Pear = glob.glob("Data_train\\Pear\\*.png")

for img_path in path_train_Carambula:
    imageCarambula = np.array(cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)).reshape(1024,1)[:,0]
    #imageCarambula = img.imread(img_path)[:, :, 0].reshape(1024, 1)[:, 0]
    #print(imageCarambula)
    images_train_Carambula.append(np.array(imageCarambula))

for img_path in path_train_Lychee:
    imageLychee = np.array(cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)).reshape(1024,1)[:,0]
    #imageLychee = img.imread(img_path)[:, :, 0].reshape(1024, 1)[:, 0]
    images_train_Lychee.append(np.array(imageLychee))

for img_path in path_train_Pear:
    imagePear = np.array(cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)).reshape(1024,1)[:,0]
    #imagePear = img.imread(img_path)[:, :, 0].reshape(1024, 1)[:, 0]
    images_train_Pear.append(np.array(imagePear))

images_train_Carambula = np.array(images_train_Carambula)/255
images_train_Lychee = np.array(images_train_Lychee)/255
images_train_Pear = np.array(images_train_Pear)/255

image_train_all = np.vstack((images_train_Carambula, images_train_Lychee, images_train_Pear))
#print(image_train_all)

pca = PCA(n_components=2)

images_train_all_compressed = pca.fit_transform(image_train_all)
images_train_all_decompressed = pca.inverse_transform(images_train_all_compressed)
images_train_all_compressed_with_label = np.hstack((images_train_all_compressed , np.array([[0]*490,[1]*490,[2]*490]).reshape(490*3,1)))

'''
def plot_c_d(image_index):
    plt.figure()
    plt.subplot(121)
    plt.imshow(images_all[image_index].reshape(32,32), cmap='gray')
    image_decompressed = pca.inverse_transform(images_all_compressed[image_index]).reshape(32,32)
    plt.subplot(122)
    plt.imshow(image_decompressed, cmap='gray')
    plt.show()
'''
class NN_homework_2_layer_so_hard(): #2_layer nn
    def __init__(self):
        self.input_size = 2     #輸入層
        self.output_size = 3    #輸出層
        self.hidden_size = 1024 #隱藏層

        np.random.seed(80)  #設置random seed以便重現結果
        rd.seed(80)         #設置random seed以便重現結果

        self.w1 = np.random.randn(self.input_size, self.hidden_size) * 0.2   #初始隨機權重
        self.w2 = np.random.randn(self.hidden_size, self.output_size) * 0.2  #初始隨機權重

        self.b1 = np.zeros(self.hidden_size)    #初始bias
        self.b2 = np.zeros(self.output_size)    #初始bias


    def relu(self, X):
        for i in range(0, X.shape[0]):
            X[i] = np.maximum(0, X[i])  #激勵函數relu function
        return X

    def _d_relu(self, z):
        for j in range(0, z.shape[0]):
            if z[j] <= 0:
                z[j] = 0
            else:
                z[j] = 1                #激勵函數relu的導數
        return z

    def softmax(self, z):
        soft = []
        z = z - max(z)
        for i in range(0, z.shape[0]):
            soft.append(np.exp(z[i]) / sum(np.exp(z)))  #softmax function
        soft = np.array(soft)
        return soft

    def forward_pass(self, X):
        self.z1 = np.dot(X, self.w1) + self.b1          #將訓練資料乘上權重再加上bias
        self.a1 = self.relu(self.z1)                    #進入激勵函數relu
        self.z2 = np.dot(self.a1, self.w2) + self.b2
        self.a2 = self.softmax(self.z2)
        #print(self.a2)
        return self.a2

    def backward_pass(self, y_pre, X, y):
        delta3 = y_pre
        #print(delta3.shape)
        delta3[int(y)] -= 1
        self.dw2 = np.dot(self.a1.reshape(self.hidden_size, 1), delta3.reshape(3, 1).T)
        #print(self.dw2)
        #print(self.dw2.shape)
        self.db2 = np.sum(delta3, axis=0)
        self.dz1 = np.dot(delta3, self.w2.T) * self._d_relu(self.a1)

        self.dw1 = np.dot(X.reshape(2, 1), self.dz1.reshape(self.hidden_size, 1).T)
        #print(self.dw1)
        #print(self.dw1.shape)
        self.db1 = np.sum(self.dz1, axis=0)

    def loss(self, y_hat, y):
        loss = -np.log2(y_hat[int(y)]) #使用cross_entropy_loss
        return loss

    def _update(self, learning_rate = 0.01): #更新權重和bias
        self.w1 = self.w1 - learning_rate * self.dw1
        self.b1 = self.b1 - learning_rate * self.db1
        self.w2 = self.w2 - learning_rate * self.dw2
        self.b2 = self.b2 - learning_rate * self.db2

    def train(self, X, iteration):
        total_loss = []
        print('training 2-layer nn:')
        for _iter in tqdm(range(iteration)):
            i = rd.randint(0, image_train_all.shape[0]-1) #隨機一筆取資料做訓練
            #print(i)
            y_hat = self.forward_pass(X[i, 0:2])
            loss = self.loss(y_hat, X[i, 2])
            #print(loss)

            self.backward_pass(y_hat, X[i, 0:2], X[i, 2])
            self._update()
            total_loss.append(loss)
            #print('iteration:' , '%d'%(_iter), end='\r')
        print('\n')
        return np.array(total_loss)

    def predict(self, X):
        pre = []

        for i in tqdm(range(0, X.shape[0])):
            y_hat = self.forward_pass(X[i, 0:2])
            pre.append(np.argmax(y_hat))

        '''
        print(X == images_train_all_compressed_with_label)
        
        if all((X == images_train_all_compressed_with_label)[0]):
            print(pre[0:10])
            print(images_train_all_compressed_with_label[0:10,2])
        else:
            print()
        '''
        return np.array(pre)

    def accurate(self, pre, y):
        current = np.sum(pre == y)
        return (current / len(y)) * 100

images_test_Carambula = []
images_test_Lychee = []
images_test_Pear = []

path_test_Carambula = glob.glob("Data_test\\Carambula\\*.png")
path_test_Lychee = glob.glob("Data_test\\Lychee\\*.png")
path_test_Pear = glob.glob("Data_test\\Pear\\*.png")

for img_path in path_test_Carambula:
    image_test_Carambula = np.array(cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)).reshape(1024, 1)[:, 0]
    #image_test_Carambula = img.imread(img_path)[:, :, 0].reshape(1024, 1)[:, 0]
    images_test_Carambula.append(np.array(image_test_Carambula))

for img_path in path_test_Lychee:
    image_test_Lychee = np.array(cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)).reshape(1024, 1)[:, 0]
    #image_test_Lychee = img.imread(img_path)[:, :, 0].reshape(1024, 1)[:, 0]
    images_test_Lychee.append(np.array(image_test_Lychee))

for img_path in path_test_Pear:
    image_test_Pear = np.array(cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)).reshape(1024, 1)[:, 0]
    #image_test_Pear = img.imread(img_path)[:, :, 0].reshape(1024, 1)[:, 0]
    images_test_Pear.append(np.array(image_test_Pear))

images_test_Carambula = np.array(images_test_Carambula)/255
images_test_Lychee = np.array(images_test_Lychee)/255
images_test_Pear = np.array(images_test_Pear)/255
images_test_all = np.vstack((images_test_Carambula, images_test_Lychee, images_test_Pear))

images_test_all_compressed = pca.transform(images_test_all)
images_test_all_decompressed = pca.inverse_transform(images_test_all_compressed)
images_test_all_compressed_with_label = np.hstack((images_test_all_compressed , np.array([[0]*166,[1]*166,[2]*166]).reshape(166*3,1)))

NN2 = NN_homework_2_layer_so_hard()
np.random.shuffle(images_train_all_compressed_with_label)
#images_train_all_compressed_with_label = np.random.shuffle(images_train_all_compressed_with_label)

episode = 10000
loss = NN2.train(images_train_all_compressed_with_label, episode)

loss_ave = []
for loss_n in range(1,len(loss)+1):
    loss_ave.append(sum(loss[0:loss_n])/loss_n)
fig, axs = plt.subplots(2, figsize=(10, 8))
axs[0].set_title('2-layer nn: loss')
axs[1].set_title('2-layer nn: average_loss')
axs[0].plot(np.array(range(0, episode)),loss)
axs[1].plot(np.array(range(0, episode)), loss_ave)
plt.show()

print('predicting train data:')
pre_train = NN2.predict(images_train_all_compressed_with_label)
time.sleep(0.1)
print("2-layer nn: train current rate: ",NN2.accurate(pre_train, images_train_all_compressed_with_label[:,2]))
print('\n')

print('predicting test data:')
pre_test = NN2.predict(images_test_all_compressed_with_label)
time.sleep(0.1)
print("2-layer nn: test current rate: ",NN2.accurate(pre_test, images_test_all_compressed_with_label[:,2]))
print('\n')

#----------------------------------------------------------------------------------------
'''
svm = SVC(C=0.5, kernel='linear')
svm.fit(images_train_all_compressed_with_label[:,0:2], images_train_all_compressed_with_label[:,2].astype(int))
plt.figure()
plot_decision_regions(images_train_all_compressed_with_label[:,0:2], images_train_all_compressed_with_label[:,2].astype(int), clf=svm, legend=2)
# Adding axes annotations
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('decision regions of training data')
plt.show()

svm.fit(images_test_all_compressed_with_label[:,0:2], images_test_all_compressed_with_label[:,2].astype(int))
plt.figure()
plot_decision_regions(images_test_all_compressed_with_label[:,0:2], images_test_all_compressed_with_label[:,2].astype(int), clf=svm, legend=2)
# Adding axes annotations
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('decision regions of testing data')
plt.show()
'''
#----------------------------------------------------------------------------------------
sample_node = 250
nx, ny = (sample_node, sample_node)
x1_min = np.min(images_test_all_compressed_with_label[:,0])
x1_max = np.max(images_train_all_compressed_with_label[:,0])
x2_min = np.min(images_train_all_compressed_with_label[:,1])
x2_max = np.max(images_train_all_compressed_with_label[:,1])

x = np.linspace(x1_min, x1_max, nx)
y = np.linspace(x2_min, x2_max, ny)
xv, yv = np.meshgrid(x, y)

xv = xv.reshape(sample_node*sample_node,1)
yv = yv.reshape(sample_node*sample_node,1)
xyv = np.hstack((xv,yv))
print('generating 2-layer nn decision regions:')
decision_regions = NN2.predict(xyv)
print('\n')

# decision regions of training data
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.contourf(xv.reshape(sample_node,sample_node), yv.reshape(sample_node,sample_node), decision_regions.reshape(sample_node,sample_node), 10, alpha=.5, cmap=plt.cm.jet)
for i in range(0,images_train_all_compressed_with_label.shape[0]):
    if images_train_all_compressed_with_label[i,2] == 0:
        kx = ax2.scatter(images_train_all_compressed_with_label[i,0], images_train_all_compressed_with_label[i,1], color = 'r')
    if images_train_all_compressed_with_label[i,2] == 1:
        yx = ax2.scatter(images_train_all_compressed_with_label[i,0], images_train_all_compressed_with_label[i,1], color = 'b')
    if images_train_all_compressed_with_label[i,2] == 2:
        zx = ax2.scatter(images_train_all_compressed_with_label[i,0], images_train_all_compressed_with_label[i,1], color = 'g')
ax2.legend(['0','1','2'])
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('2-layer nn: decision regions of training data')

# decision regions of testing data
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.contourf(xv.reshape(sample_node,sample_node), yv.reshape(sample_node,sample_node), decision_regions.reshape(sample_node,sample_node), 10, alpha=.5, cmap=plt.cm.jet)
for i in range(0,images_test_all_compressed_with_label.shape[0]):
    if images_test_all_compressed_with_label[i,2] == 0:
        kx = ax2.scatter(images_test_all_compressed_with_label[i,0], images_test_all_compressed_with_label[i,1], color = 'r')
    if images_test_all_compressed_with_label[i,2] == 1:
        yx = ax2.scatter(images_test_all_compressed_with_label[i,0], images_test_all_compressed_with_label[i,1], color = 'b')
    if images_test_all_compressed_with_label[i,2] == 2:
        zx = ax2.scatter(images_test_all_compressed_with_label[i,0], images_test_all_compressed_with_label[i,1], color = 'g')
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('2-layer nn: decision regions of testing data')

#----------------------------------------------------------------------------------------
class NN_homework_3_layer_so_hard():  # 3_layer nn
    def __init__(self):
        self.input_size = 2  # 輸入層
        self.output_size = 3  # 輸出層
        self.hidden_size_1 = 256  # 隱藏層
        self.hidden_size_2 = 128  # 隱藏層

        np.random.seed(80)  # 設置random seed以便重現結果
        rd.seed(80)  # 設置random seed以便重現結果

        self.w1 = np.random.randn(self.input_size, self.hidden_size_1) * 0.2  # 初始隨機權重
        self.w2 = np.random.randn(self.hidden_size_1, self.hidden_size_2) * 0.2  # 初始隨機權重
        self.w3= np.random.randn(self.hidden_size_2, self.output_size) * 0.2  # 初始隨機權重
        self.b1 = np.zeros(self.hidden_size_1)  # 初始bias
        self.b2 = np.zeros(self.hidden_size_2)  # 初始bias
        self.b3 = np.zeros(self.output_size)    # 初始bias

    def relu(self, X):
        for i in range(0, X.shape[0]):
            X[i] = np.maximum(0, X[i])  # 激勵函數relu function
        return X

    def _d_relu(self, z):
        for j in range(0, z.shape[0]):
            if z[j] <= 0:
                z[j] = 0
            else:
                z[j] = 1  # 激勵函數relu的導數
        return z

    def softmax(self, z):
        soft = []
        z = z - max(z)
        for i in range(0, z.shape[0]):
            soft.append(np.exp(z[i]) / sum(np.exp(z)))  # softmax function
        soft = np.array(soft)
        return soft

    def forward_pass(self, X):
        self.z1 = np.dot(X, self.w1) + self.b1  # 將訓練資料乘上權重再加上bias
        self.a1 = self.relu(self.z1)  # 進入激勵函數relu
        self.z2 = np.dot(self.a1, self.w2) + self.b2
        self.a2 = self.relu(self.z2)
        self.z3 = np.dot(self.a2, self.w3) + self.b3
        self.a3 = self.softmax(self.z3)
        return self.a3

    def backward_pass(self, y_pre, X, y):
        delta3 = y_pre

        delta3[int(y)] -= 1
        self.dw3 = np.dot(self.a2.reshape(self.hidden_size_2, 1), delta3.reshape(3, 1).T)
        self.db3 = np.sum(delta3, axis=0)

        self.dz2 = np.dot(self.w3, delta3.T) * self._d_relu(self.a2)
        self.dw2 = np.dot(self.a1.reshape(self.hidden_size_1, 1), self.dz2.reshape(self.hidden_size_2, 1).T)
        self.db2 = np.sum(self.dz2, axis=0)

        self.dz1 = np.dot(self.w2, self.dz2) * self._d_relu(self.a1)
        self.dw1 = np.dot(X.reshape(2, 1), self.dz1.reshape(self.hidden_size_1, 1).T)
        self.db1 = np.sum(self.dz1, axis=0)

    def loss(self, y_hat, y):
        loss = -np.log2(y_hat[int(y)])  # 使用cross_entropy_loss
        return loss

    def _update(self, learning_rate=0.01):  # 更新權重和bias
        self.w1 = self.w1 - learning_rate * self.dw1
        self.b1 = self.b1 - learning_rate * self.db1
        self.w2 = self.w2 - learning_rate * self.dw2
        self.b2 = self.b2 - learning_rate * self.db2
        self.w3 = self.w3 - learning_rate * self.dw3
        self.b3 = self.b3 - learning_rate * self.db3

    def train(self, X, iteration):
        total_loss = []
        print('training 3-layer nn:')
        for _iter in tqdm(range(iteration)):
            i = rd.randint(0, image_train_all.shape[0] - 1)  # 隨機一筆取資料做訓練
            # print(i)
            y_hat = self.forward_pass(X[i, 0:2])
            loss = self.loss(y_hat, X[i, 2])
            # print(loss)

            self.backward_pass(y_hat, X[i, 0:2], X[i, 2])
            self._update()
            total_loss.append(loss)
            # print('iteration:' , '%d'%(_iter), end='\r')
        print('\n')
        return np.array(total_loss)

    def predict(self, X):
        pre = []

        for i in tqdm(range(0, X.shape[0])):
            y_hat = self.forward_pass(X[i, 0:2])
            pre.append(np.argmax(y_hat))

        '''
        print(X == images_train_all_compressed_with_label)

        if all((X == images_train_all_compressed_with_label)[0]):
            print(pre[0:10])
            print(images_train_all_compressed_with_label[0:10,2])
        else:
            print()
        '''
        return np.array(pre)

    def accurate(self, pre, y):
        current = np.sum(pre == y)
        return (current / len(y)) * 100

NN3 = NN_homework_3_layer_so_hard()
np.random.shuffle(images_train_all_compressed_with_label)
#images_train_all_compressed_with_label = np.random.shuffle(images_train_all_compressed_with_label)

episode = 10000
loss = NN3.train(images_train_all_compressed_with_label, episode)

loss_ave = []
for loss_n in range(1,len(loss)+1):
    loss_ave.append(sum(loss[0:loss_n])/loss_n)
fig, axs = plt.subplots(2, figsize=(10, 8))
axs[0].set_title('3-layer nn: loss')
axs[1].set_title('3-layer nn: average_loss')
axs[0].plot(np.array(range(0, episode)),loss)
axs[1].plot(np.array(range(0, episode)), loss_ave)
plt.show()

print('predicting train data:')
pre_train = NN3.predict(images_train_all_compressed_with_label)
time.sleep(0.1)
print("3-layer nn: train current rate: ",NN3.accurate(pre_train, images_train_all_compressed_with_label[:,2]))
print('\n')

print('predicting test data:')
pre_test = NN3.predict(images_test_all_compressed_with_label)
time.sleep(0.1)
print("3-layer nn: test current rate: ",NN3.accurate(pre_test, images_test_all_compressed_with_label[:,2]))
print('\n')

sample_node = 250
nx, ny = (sample_node, sample_node)
x1_min = np.min(images_test_all_compressed_with_label[:,0])
x1_max = np.max(images_train_all_compressed_with_label[:,0])
x2_min = np.min(images_train_all_compressed_with_label[:,1])
x2_max = np.max(images_train_all_compressed_with_label[:,1])

x = np.linspace(x1_min, x1_max, nx)
y = np.linspace(x2_min, x2_max, ny)
xv, yv = np.meshgrid(x, y)

xv = xv.reshape(sample_node*sample_node,1)
yv = yv.reshape(sample_node*sample_node,1)
xyv = np.hstack((xv,yv))
print('generating 3-layer nn decision regions:')
decision_regions = NN3.predict(xyv)
print('\n')

# decision regions of training data
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.contourf(xv.reshape(sample_node,sample_node), yv.reshape(sample_node,sample_node), decision_regions.reshape(sample_node,sample_node), 10, alpha=.5, cmap=plt.cm.jet)
for i in range(0,images_train_all_compressed_with_label.shape[0]):
    if images_train_all_compressed_with_label[i,2] == 0:
        kx = ax2.scatter(images_train_all_compressed_with_label[i,0], images_train_all_compressed_with_label[i,1], color = 'r')
    if images_train_all_compressed_with_label[i,2] == 1:
        yx = ax2.scatter(images_train_all_compressed_with_label[i,0], images_train_all_compressed_with_label[i,1], color = 'b')
    if images_train_all_compressed_with_label[i,2] == 2:
        zx = ax2.scatter(images_train_all_compressed_with_label[i,0], images_train_all_compressed_with_label[i,1], color = 'g')
ax2.legend(['0','1','2'])
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('3-layer nn: decision regions of training data')

# decision regions of testing data
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.contourf(xv.reshape(sample_node,sample_node), yv.reshape(sample_node,sample_node), decision_regions.reshape(sample_node,sample_node), 10, alpha=.5, cmap=plt.cm.jet)
for i in range(0,images_test_all_compressed_with_label.shape[0]):
    if images_test_all_compressed_with_label[i,2] == 0:
        kx = ax2.scatter(images_test_all_compressed_with_label[i,0], images_test_all_compressed_with_label[i,1], color = 'r')
    if images_test_all_compressed_with_label[i,2] == 1:
        yx = ax2.scatter(images_test_all_compressed_with_label[i,0], images_test_all_compressed_with_label[i,1], color = 'b')
    if images_test_all_compressed_with_label[i,2] == 2:
        zx = ax2.scatter(images_test_all_compressed_with_label[i,0], images_test_all_compressed_with_label[i,1], color = 'g')
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('3-layer nn: decision regions of testing data')
