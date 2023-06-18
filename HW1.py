import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

#2-1----------------------------------------------------------------------------------------
filename = 'wine.csv'
df = pd.read_csv(filename) # 讀取wine.csv

result30_array = [] # 建立儲存每次分類準確度的空白陣列
for t in range(30): # 分類30次以觀察平均準確度
    for i in [0,1,2]:
        locals()['df_target'+str(i)] = pd.DataFrame(columns=list(df.columns)) # 建立分別儲存三個target的空白表單
        locals()['df_target'+str(i)] = df.loc[df['target'] == i] # 複製三個target的資料
        locals()['df_target'+str(i)] = locals()['df_target'+str(i)].reindex(np.random.permutation(locals()['df_target'+str(i)].index)) # 亂數排列
        locals()['df_target'+str(i)].reset_index(drop=True, inplace=True) # 重新排序index

    df_train = pd.DataFrame(columns=list(df.columns)) #建立訓練集的空白表單
    df_test = pd.DataFrame(columns=list(df.columns)) #建立測試集的空白表單

    df_test = pd.concat([df_target0.loc[0:19,:],df_target1.loc[0:19,:],df_target2.loc[0:19,:]]) # 從三個亂數過的target表單，各複製20筆資料，做為測試集
    df_test.reset_index(drop=True, inplace=True) # 重新排序index
    df_train = pd.concat([df_target0.loc[20:len(df_target0),:],df_target1.loc[20:len(df_target1),:],df_target2.loc[20:len(df_target2),:]]) # 從三個亂數過的target表單，取勝下的資料，做為訓練集
    df_train.reset_index(drop=True, inplace=True) # 重新排序index

    df_train.to_csv('train.csv') # 儲存訓練集
    df_test.to_csv('test.csv') # 儲存測試集

#2-2----------------------------------------------------------------------------------------
    # 計算各個feature的平均值與變異數
    train_mean_array = [[],[],[]] # 建立三個target的各features平均值陣列
    train_var_array = [[],[],[]] # 建立三個target的各features變異數陣列
    for i in range(3):
        for j in range(1,14,1):
            train_mean = np.mean(df_train.loc[df_train['target'] == i, df_train.columns[j]]) # 計算三個target的各features的平均值
            train_mean_array[i].append(train_mean) # 平均值放進平均值陣列
            train_var = np.var(df_train.loc[df_train['target'] == i, df_train.columns[j]]) # 計算三個target的各features的變異數
            train_var_array[i].append(train_var) # 變異數放進變異數陣列

    # 計算事前機率(priori probability)
    for i in [0,1,2]:
        locals()['train_target' + str(i) + '_priori_probability'] = len(df_train.loc[df_train['target'] == i])/len(df_train)

    # 計算概似函數(likelihood)與後驗機率(posterior probability)
    test_result = []
    for j in range(len(df_test)): # 讀取測試集的每一個rows
        likelihood0 = 1 * train_target0_priori_probability # 後驗機率的分子預先乘上事前機率
        likelihood1 = 1 * train_target1_priori_probability # 後驗機率的分子預先乘上事前機率
        likelihood2 = 1 * train_target2_priori_probability # 後驗機率的分子預先乘上事前機率
        #print(j)
        for i in range(13):
            likelihood0 = likelihood0 * ((1 / (np.sqrt(2 * np.pi * train_var_array[0][i]))) * np.exp(-((df_test.loc[j,df_test.columns[i+1]] - train_mean_array[0][i]) ** 2) / (2 * (train_var_array[0][i])))) # 計算各features的likelihood相乘到後驗機率的分子
            likelihood1 = likelihood1 * ((1 / (np.sqrt(2 * np.pi * train_var_array[1][i]))) * np.exp(-((df_test.loc[j,df_test.columns[i+1]] - train_mean_array[1][i]) ** 2) / (2 * (train_var_array[1][i])))) # 計算各features的likelihood相乘到後驗機率的分子
            likelihood2 = likelihood2 * ((1 / (np.sqrt(2 * np.pi * train_var_array[2][i]))) * np.exp(-((df_test.loc[j,df_test.columns[i+1]] - train_mean_array[2][i]) ** 2) / (2 * (train_var_array[2][i])))) # 計算各features的likelihood相乘到後驗機率的分子
            #print(likelihood0)
            #print('t:',df_test.loc[j,df_test.columns[i+1]])

        likelihood_max = np.argmax([likelihood0,likelihood1,likelihood2]) # 選擇三個後驗機率中的最大值，作為分類的解答
        test_result.append(likelihood_max) # 寫進分類的解答陣列

    c = 0
    for i in range(60):
        if df_test.loc[i,'target'] == test_result[i]:
            c += 1 # 統計測試集中分類正確的數量
    print('Accuracy rate：', round(c/60*100, 2), '%') #print出分類正確的比率
    result30_array.append(round(c/60*100, 2))

print('------------------------------------\n平均準確度： ', round(np.average(result30_array),2), '%')  #print出平均準確度

#2-3----------------------------------------------------------------------------------------
df_test_list = df_test.values.tolist() # 將測試集的表單轉為list
pca = PCA(n_components = 3) # 將資料降為三維
pca.fit(df_test_list) # fit出特徵映射後使資料變異量最大的投影向量
test_pca = pca.transform(df_test_list)
ax = plt.figure().add_subplot(projection = '3d') #將figure設定為三維
ax.scatter(test_pca[:20,0], test_pca[:20,1], test_pca[:20,2],color = 'r')
ax.scatter(test_pca[20:40,0], test_pca[20:40,1], test_pca[20:40,2],color = 'g')
ax.scatter(test_pca[40:60,0], test_pca[40:60,1], test_pca[40:60,2],color = 'b')
plt.title('Classification result of testing data \n Accuracy rate: ' + str(round(c/60*100,2)) + '%')

#2-4----------------------------------------------------------------------------------------
