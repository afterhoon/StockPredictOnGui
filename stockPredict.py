
# coding: utf-8

# In[ ]:


import tensorflow as tf
import numpy as np
import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import pandas_datareader.data as data
import pandas as pd
from pandas import Series, DataFrame
import fix_yahoo_finance as yf
import importlib
from datetime import date, timedelta

class MyWindow(QWidget):
    def calcStock(self):
        tf.reset_default_graph()
        tf.set_random_seed(777)

        def data_standardization(x):
            x_np = np.asarray(x)
            return (x_np - x_np.mean()) / x_np.std()


        def min_max_scaling(x):
            x_np = np.asarray(x)
            return (x_np - x_np.min()) / (x_np.max() - x_np.min() + 1e-7) 

        def reverse_min_max_scaling(org_x, x):
            org_x_np = np.asarray(org_x)
            x_np = np.asarray(x)
            return (x_np * (org_x_np.max() - org_x_np.min() + 1e-7)) + org_x_np.min()
        

        input_data_column_cnt = 6
        output_data_column_cnt = 1

        seq_length = 28
        rnn_cell_hidden_dim = 20
        forget_bias = 1.0   
        num_stacked_layers = 1   
        keep_prob = 1.0   

        epoch_num = 50       
        learning_rate = 0.01

        yf.pdr_override()
        start_date = '1996-05-06'
        print(start_date)
        print(self.comCode)
        df = data.get_data_yahoo(self.comCode, start_date)
        df.to_csv('./STOCK.csv', mode='w') 
        stock_file_name = 'STOCK.csv' 
        encoding = 'euc-kr' 
        names = ['Date','Open','High','Low','Close','Adj Close','Volume']

        raw_dataframe = pd.read_csv(stock_file_name, names=names, encoding=encoding) 
        del raw_dataframe['Date']
        stock_info = raw_dataframe.values[1:].astype(np.float) 

        price = stock_info[:,:-1]
        norm_price = min_max_scaling(price) 

        volume = stock_info[:,-1:]
        norm_volume = min_max_scaling(volume) 

        x = np.concatenate((norm_price, norm_volume), axis=1) 
        y = x[:, [-2]] 

        dataX = []
        dataY = [] 

        for i in range(0, len(y) - seq_length):
            _x = x[i : i+seq_length]
            _y = y[i + seq_length] 
            if i is 0:
                 print(_x, "->", _y) 
            dataX.append(_x) 
            dataY.append(_y) 

        train_size = int(len(dataY) * 0.7)
        test_size = len(dataY) - train_size

        trainX = np.array(dataX[0:train_size])
        trainY = np.array(dataY[0:train_size])

        testX = np.array(dataX[train_size:len(dataX)])
        testY = np.array(dataY[train_size:len(dataY)])

        X = tf.placeholder(tf.float32, [None, seq_length, input_data_column_cnt])
        Y = tf.placeholder(tf.float32, [None, 1])

        targets = tf.placeholder(tf.float32, [None, 1])
        predictions = tf.placeholder(tf.float32, [None, 1])

        def lstm_cell():
            cell = tf.contrib.rnn.BasicLSTMCell(num_units=rnn_cell_hidden_dim, 
                                        forget_bias=forget_bias, state_is_tuple=True, activation=tf.nn.softsign)
            if keep_prob < 1.0:
                cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)
            return cell

        stackedRNNs = [lstm_cell() for _ in range(num_stacked_layers)]
        multi_cells = tf.contrib.rnn.MultiRNNCell(stackedRNNs, state_is_tuple=True) if num_stacked_layers > 1 else lstm_cell()

        hypothesis, _states = tf.nn.dynamic_rnn(multi_cells, X, dtype=tf.float32)
        print("hypothesis: ", hypothesis)

        hypothesis = tf.contrib.layers.fully_connected(hypothesis[:, -1], output_data_column_cnt, activation_fn=tf.identity)

        loss = tf.reduce_sum(tf.square(hypothesis - Y))
        optimizer = tf.train.AdamOptimizer(learning_rate)

        train = optimizer.minimize(loss)

        rmse = tf.sqrt(tf.reduce_mean(tf.squared_difference(targets, predictions)))

        train_error_summary = [] 
        test_error_summary = []  
        test_predict = ''  

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        for epoch in range(epoch_num):
            _, _loss = sess.run([train, loss], feed_dict={X: trainX, Y: trainY})
            if ((epoch+1) % 100 == 0) or (epoch == epoch_num-1): 
                train_predict = sess.run(hypothesis, feed_dict={X: trainX})
                train_error = sess.run(rmse, feed_dict={targets: trainY, predictions: train_predict})
                train_error_summary.append(train_error)

                test_predict = sess.run(hypothesis, feed_dict={X: testX})
                test_error = sess.run(rmse, feed_dict={targets: testY, predictions: test_predict})
                test_error_summary.append(test_error)
                
        recent_data = np.array([x[len(x)-seq_length : ]])
        print("recent_data.shape:", recent_data.shape)
        print("recent_data:", recent_data)

        price_predict = sess.run(hypothesis, feed_dict={X: recent_data})

        print("test_predict", test_predict[0])
        
        print("--------------------------------------------------")
        for i in range(10):
            print(test_predict[0])
        print("--------------------------------------------------")
        
        price_predict = reverse_min_max_scaling(price,price_predict) 
        test_predict = reverse_min_max_scaling(price,test_predict)
        testY = reverse_min_max_scaling(price,testY) 
        print("Tomorrow's stock price", price_predict[0]) 
        
        return testY, test_predict, train_error_summary, test_error_summary, price_predict[0]
    
    def __init__(self):
        super().__init__()
        self.setupUI()

    def setupUI(self):
        self.setGeometry(600, 200, 1200, 600)
        self.setWindowTitle("PyChart Viewer v0.1")
        self.setWindowIcon(QIcon('icon.png'))

        self.comboBox = QComboBox()
        self.comboBox.addItems(['Kia 모터스', '네이버', 'SK hynix', 'LG전자', '다음 카카오','아마존'])
        self.okButton = QPushButton("확인")
        self.okButton.clicked.connect(self.okButtonClicked)
        
        self.pushButton = QPushButton("가격 차트")
        self.pushButton.clicked.connect(self.pushButtonClicked)
        
        self.pushButton2 = QPushButton("학습 차트")
        self.pushButton2.clicked.connect(self.pushButtonClicked2)
        
        self.company = "Company"
        self.comCode = ""
        self.price = str(0)
        
        self.resData = []
        
        self.comLa = QLabel(self.company + "의") 
        self.priceLa = QLabel(self.price + "원") 

        self.fig = plt.Figure()
        self.canvas = FigureCanvas(self.fig)

        leftLayout = QVBoxLayout()
        leftLayout.addWidget(self.canvas)

        # Right Layout
        rightLayout = QVBoxLayout()
        rightLayout.addWidget(self.comboBox)
        rightLayout.addWidget(self.okButton)
        rightLayout.addWidget(self.pushButton)
        rightLayout.addWidget(self.pushButton2)
        rightLayout.addWidget(QLabel(""))
        rightLayout.addWidget(self.comLa)
        rightLayout.addWidget(QLabel("내일 예상 주가"))
        rightLayout.addWidget(self.priceLa)
        rightLayout.addStretch(1) 

        layout = QHBoxLayout()
        layout.addLayout(leftLayout)
        layout.addLayout(rightLayout)
        layout.setStretchFactor(leftLayout, 1)
        layout.setStretchFactor(rightLayout, 0) 

        self.setLayout(layout)

    def okButtonClicked(self):
        tf.reset_default_graph()
        code = ['000270.KS','035420.KS','000660.KS','066570.KS','035720.KS','AMZN']
        self.company = self.comboBox.currentText()
        self.comCode = code[self.comboBox.currentIndex()]
        print("this: " + self.company + "(" + self.comCode + ")")

        self.resData = self.calcStock()
        print("-------------------------------------------------------------------")
        for i in range(5):
            print("self.resData[" + str(i) + "]")
            print(self.resData[i])
        print("-------------------------------------------------------------------")
        self.comLa.setText(self.company + "의")
        self.priceLa.setText(str(self.resData[4]) + "원")
        
    def pushButtonClicked(self):
        self.fig.clf()

        ax = self.fig.add_subplot(111)
        ax.plot(self.resData[0], 'r', label="Price") # testY
        ax.plot(self.resData[1], 'b', label="Predict") # test_predict
        ax.legend(loc='upper right')
        ax.grid()

        self.canvas.draw()
        
    def pushButtonClicked2(self):
        self.fig.clf()

        ax = self.fig.add_subplot(111)
        
        ax.plot(self.resData[2], 'gold', label="train_error") # train_error_summary
        ax.plot(self.resData[3], 'b', label="test_error") # test_error_summary
        ax.legend(loc='upper right')
        ax.grid()

        self.canvas.draw()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MyWindow()
    window.show()
    app.exec_()
