#!/usr/bin/env python
# coding: utf-8

# In[26]:


import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler


# Hyper parameters

# In[27]:


WINDOW_SIZE=10
BATCH_SIZE=32

class StockTransformer(nn.Module):
        def __init__(self,input_dim,d_model,nhead,num_layers,dropout):
            super().__init__()
            self.input_dim=input_dim
            self.d_model=d_model
            #將輸入維度轉為d_model
            self.feature_embedding = nn.Linear(input_dim, d_model)
            #加入positional encoding(用learnable para)
            self.positional_encoding=nn.Parameter(torch.rand(1,10,d_model))
            #Encoder
            encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            #輸出回歸層
            self.regressor = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.ReLU(),
                nn.Linear(d_model // 2, 1)
            )
        def forward(self,x):
            x=self.feature_embedding(x)
            x=x+self.positional_encoding
            encoded=self.encoder(x)
            last=encoded[:,-1,:]
            output=self.regressor(last)

            return output

# 股票資料使用從2025-02-14到2025-05-16共62天(但實際只有61天,因為每天的明天要作為當天的ground truth,所以最後一天20250516刪除只做為0517的label)

# In[3]:
if __name__ == "__main__":

    # 讀取股票資料
    stock_data = pd.read_csv('./tmp/stock_data.csv', dtype={"stock_code": str, "date": str})
    stock_data.set_index(["stock_code", "date"], inplace=True)
    stock_data.sort_index(inplace=True)


    # 訓練輸入股票特徵選用6種並對其資料做Z-score標準化

    # In[4]:


    stocks_data_frame=[]
    scaler = StandardScaler()

    # 定義訓練資料集
    for stock_code in stock_data.index.get_level_values(0).unique():
        stock_df = pd.DataFrame({
            stock_code+'open': stock_data.loc[stock_code]['open'],#開盤價
            stock_code+'close': stock_data.loc[stock_code]['close'],#收盤價'
            stock_code+'high': stock_data.loc[stock_code]['high'],#最高價
            stock_code+'low': stock_data.loc[stock_code]['low'],#最低價
            stock_code+'volume': stock_data.loc[stock_code]['transaction_volume'],#成交量
            stock_code+'change': stock_data.loc[stock_code]['change'],#漲跌幅
        })
        
        # 對數值col進行Z-score標準化
        numeric_columns = [stock_code+'open', stock_code+'close', stock_code+'high', stock_code+'low', stock_code+'volume', stock_code+'change']
        scaler.fit(stock_df[numeric_columns])
        stock_df[numeric_columns] = scaler.transform(stock_df[numeric_columns])
        
        
        stocks_data_frame.append(stock_df)


    # 刪除那些在這段期間沒有全部有資料(也就是不足62天)的股票=>所剩1574檔股票

    # In[5]:


    # 檢查並刪除不符合(62,4)大小的資料框
    abnormal_stocks = []
    for i in range(len(stocks_data_frame)):
        if stocks_data_frame[i].shape != (62, 6):
            abnormal_stocks.append(i)

    #將原本資料刪除這些不完整的股票
    stocks_data_frame = [df for i, df in enumerate(stocks_data_frame) if i not in abnormal_stocks]

    print(f"刪除後剩餘股票數量: {len(stocks_data_frame)}")


    # 將所有股票資料同一天的concat到同一row=>shape=(天數,股票數量*特徵數)
    # 
    # 使模型能夠學習不同股票間的關連性

    # 替剩餘每支股票建立一個整數ID供之後模型embedding輸入

    # In[182]:


    # # 從list中取得所有不重複的股票代碼
    # ids = []
    # for df in stocks_data_frame:
    #     code = df['stock_code'].iloc[0]#取得資料中所有stock code
    #     if code not in ids:
    #         ids.append(code)
    # stock_code_to_id = {code: idx for idx, code in enumerate(ids)}#建立一個映射表


    # In[ ]:


    # print(stock_code_to_id)


    # In[177]:


    # # 刪除 stock_code 欄位
    # for df in stocks_data_frame:
    #     df.drop('stock_code', axis=1, inplace=True)


    # In[118]:


    # #將剩餘股票中的股票代碼用獨立整數id替代
    # for df in stocks_data_frame:
    #     df['stock_code'] = df['stock_code'].map(stock_code_to_id)


    # 將每支股票資料往前移一天作為label供訓練,並只取change作訓練

    # In[7]:


    #將所有股票往前一天移動來產生預測label
    #因為最後只要預測漲跌幅,所有change_labels只儲存labels中change的col
    labels = []
    shifted_labels=[]
    shifted_stocks = []

    for stock_df in stocks_data_frame:
        # 產生label
        labels = stock_df.shift(-1)
        labels = labels.iloc[:-1]  # 刪除最後一天(NaN)
        
        # 只取每5個column的change作為label
        change_cols = [col for col in labels.columns if 'change' in col]
        labels = labels[change_cols]
        shifted_labels.append(labels)
        
        # 刪除最後一天資料與label統一
        new_df = stock_df.iloc[:-1]
        shifted_stocks.append(new_df)



    # In[72]:


    shifted_labels[-1]


    # 使用sliding windows來處理data,設定window size=10(因input資料只有61天)

    # In[12]:


    def create_seq_data(stocks,labels,seq_size):
        seq=[]#裝每支股票經過windows切分過後的資料
        seq_label=[]#裝每支股票中每個window的label

        for stock,label in zip(stocks,labels):
            for i in range(len(stock)-seq_size+1):
                seq.append(stock.iloc[i:i+seq_size])
                seq_label.append(label.iloc[i+seq_size-1])

        return np.array(seq), np.array(seq_label)


    # 用每10天作為一筆資料並預測第11天的change來將61天的資料切分data和Label為時間序列模型可輸入的資料

    # In[13]:


    x,y=create_seq_data(shifted_stocks,shifted_labels,WINDOW_SIZE)


    # x(Data) shape:(每支股票切分後的window數*有幾檔股票,window size,特徵數)
    # 
    # y(Label) shape:(每支股票切分後的window數*有幾檔股票)

    # In[25]:


    print(x.shape)
    print(y.shape)


    # 將以上資料切分成訓練集和測試集並準備Data Loader

    # In[66]:


    from sklearn.model_selection import train_test_split

    X_train, X_val, y_train, y_val = train_test_split(
        x, 
        y,
        test_size=0.2,#驗證集佔20%
        shuffle=False  
    )

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32)



    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)



    # 準備模型(使用Encoder-only 的Transformer)

    # In[59]:


   


    # In[60]:


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)


    # 建立模型以及設定參數

    # In[61]:


    model = StockTransformer(input_dim=6,d_model=128 ,nhead=2,num_layers=2,dropout=0.1).to(device)


    # In[62]:


    print(model)


    # 設定Loss function和optimizer
    # 

    # In[63]:


    criterion = nn.MSELoss()  
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


    # In[64]:


    def train_epoch(model, loader, optimizer, criterion):
        model.train()
        total_loss = 0
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device).float()   # (B, 10, 6)
            y_batch = y_batch.to(device).float()   # (B, 1)

            optimizer.zero_grad()
            preds = model(X_batch)                # shape: (B, 1)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * X_batch.size(0)
        return total_loss / len(loader.dataset)

    def eval_epoch(model, loader, criterion):
        model.eval()
        total_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(device).float()
                y_batch = y_batch.to(device).float()
                preds = model(X_batch)
                loss = criterion(preds, y_batch)
                total_loss += loss.item() * X_batch.size(0)
        return total_loss / len(loader.dataset)


    # 訓練並記錄訓練結果和儲存模型

    # In[67]:


    EPOCHS = 20

    train_losses=[]
    val_losses = []

    for epoch in range(EPOCHS):
        train_loss = train_epoch(model, train_loader, optimizer, criterion)
        val_loss = eval_epoch(model, val_loader, criterion)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
    #儲存模型
    torch.save(model.state_dict(), "stock_transformer.pth")


    # In[68]:


    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss', marker='o')
    plt.plot(val_losses, label='Validation Loss', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Training and Validation Loss Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


    # Prediction

    # In[85]:


    model = StockTransformer(input_dim=6, d_model=128, nhead=4, num_layers=2, dropout=0.1)
    model.load_state_dict(torch.load("stock_transformer.pth"))
    model = model.to(device)  # 將模型移到GPU
    model.eval()


    # In[102]:


    import random
    random_number = random.randint(0,len(X_val))



    x_test=X_val[random_number]
    y_test=y_val[random_number]

    target = x_test
    x_input = torch.tensor(target).float().unsqueeze(0)  # shape: (1, 10, 6)
    x_input = x_input.to(device)  # 將輸入數據移到GPU

    with torch.no_grad():
        prediction = model(x_input)
        print(f"預測漲跌幅:{prediction.item()}")  
        print(f"實際漲跌幅:{y_test.item()}")
        print("\n預測與實際漲跌趨勢是否相同:", (prediction.item() > 0 and y_test.item() > 0) or (prediction.item() < 0 and y_test.item() < 0))