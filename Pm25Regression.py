#!/home/pengwei/anaconda2/bin/python
#encoding:utf-8
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model

def main():
    train = pd.read_csv('./data/train.csv')
    test = pd.read_csv('./data/test_X.csv')
    # print test.head(2)
    train = train.drop(['Date', 'Location'], 1)
    # print train.head(2)
    # print train['Observation'] == 'PM2.5' # True
    # 0 False
    # 1 False
    # 2 False
    # 把训练数据集进行筛选
    pm25 = train[train['Observation'] == 'PM2.5']
    # print pm25
    # 9 PM2.5 15 25 25 ... 42 49
    pm25 = pm25.drop(['Observation'], 1)
    # drop can not change the original object
    # print pm25.head(3)
    # print pm25.shape   (240, 24)
    dfList = []
    # s1 = pd.Series([0, 1, 2], index=['a', 'b', 'c'])
    # dfList.append(s1)
    # print dfList
    # s2 = pd.Series([2, 3, 4], index=['c', 'f', 'e'])
    # dfList.append(s2)
    # print '2', dfList
    # print pd.concat(dfList)

    # pm25的shape 240 × 24, 可以做15次这样的筛选，然后再把这15次的筛选连接起来
    # 0-9, 1-10....14-23    (240*15) * 10的数据集
    for i in range(15):
        df = pm25.iloc[:, i:i + 10]
        df.columns = np.array(range(10))
        dfList.append(df)
        pass
    pm25 = pd.concat(dfList)
    # 3600 * 0.8 = 2880用于训练   720用于测试
    pm25_x = pm25.iloc[0:2880, 0:9]
    # print pm25_x
    pm25_xpre = pm25.iloc[2880:3600, 0:9]
    # print pm25_x.shape  #(2880,9)
    # print pm25_xpre.shape  # (720,9)
    pm25_y = pm25.iloc[0:2880, 9:10]
    pm25_ypre = pm25.iloc[2880:3600, 9:10]
    # print pm25_y.shape  #(2880,1)
    # print pm25_y
    pm25.to_csv('./data/pm25.csv')


    # 对测试数据集开始筛选
    pm25Test = test[test['Observation'] == 'PM2.5']
    # 删除掉多余的属性
    pm25Test = pm25Test.drop(['Date', 'Observation'], 1)

    pm25Test = pm25Test.iloc[:, 1:10]
    pm25Test.to_csv("./data/pm25Test.csv")
    # 多出了一列 0,9,27,45....
    # print pm25Test.shape (240,9)
    result = linear_model_main(pm25_x, pm25_y, pm25_xpre)
    print "Intercept value ", result['intercept']
    print "coefficient", result['coefficient']
    # print "Predicted value: ", result['predicted_value']
    show_linear_line(pm25_xpre, pm25_ypre)
    pass
# 现在让我们把X_parameter和Y_parameter拟合为线性回归模型。
# 我们要写一个函数，输入为X_parameters、Y_parameter和你要预测的平方英尺值，返回θ0、θ1和预测出的价格值。
def linear_model_main(pm25_x, pm25_y, pm25Test):
    # Creat linear regression object
    regr = linear_model.LinearRegression()
    regr.fit(pm25_x, pm25_y)
    predict = regr.predict(pm25Test)
    # 我们创建一个名称为predictions的字典，存着θ0、θ1和预测值，并返回predictions字典为输出
    predictions = {}
    predictions['intercept'] = regr.intercept_
    predictions['coefficient'] = regr.coef_
    predictions['predicted_value'] = predict
    return predictions

# draw
def show_linear_line(X_parameters, Y_parameters):
    # Create linear regression object
    # (720,9)  (720,1)
    regr = linear_model.LinearRegression()
    regr.fit(X_parameters, Y_parameters)
    predict = regr.predict(X_parameters)  # (720,1)
    # 合并两个矩阵
    xy_real = np.hstack((X_parameters, Y_parameters))
    xy_pre = np.hstack((X_parameters, predict))
    # print xy_merge.shape
    x = np.array(range(10))
    i = 1
    while True:
        y = xy_real[i]
        y_pred = xy_pre[i]
        plt.scatter(x, y, color='blue')
        plt.plot(x, y_pred, color='red', linewidth=4)
        i += 1
        if i > 10:
            break
        pass

    plt.xticks(())
    plt.yticks(())
    plt.show()
    pass
# def show_linear_line(X_parameters,Y_parameters):
#     # Create linear regression object
#     regr = linear_model.LinearRegression()
#     regr.fit(X_parameters, Y_parameters)
#     plt.scatter(X_parameters,Y_parameters,color='blue')
#     plt.plot(X_parameters,regr.predict(X_parameters),color='red',linewidth=4)
#     plt.xticks(())
#     plt.yticks(())
#     plt.show()


if __name__ == '__main__':
    main()