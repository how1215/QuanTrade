import torch
import joblib
import os
import numpy as np

from datetime import datetime, timedelta
from Stock_API import Stock_API
from train_1 import StockTransformer

current_dir = os.path.dirname(os.path.abspath(__file__))
scaler_path = os.path.join(current_dir, './model/scaler.pkl')
#用戶登入
user = Stock_API('P76131717' , 'P76131717')
#預測模型權重路徑
model_dir='./model/stock_transformer.pth'
#設定設備
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#用前10天資料預測
time_window=13
#30支特定股票池(全部股票太多所以只選特定股票做預測)
tech_stocks = [
    "2330",  # 台積電
    "2317",  # 鴻海
    "2454",  # 聯發科
    "2308",  # 台達電
    "2379",  # 瑞昱
    "2382",  # 廣達 
    "3008",  # 大立光
    "3711",  # 日月光投控
    "2303",  # 聯電
    "2408",  # 南亞科
    "2337",  # 旺宏
    "2344",  # 華邦電
    "2383",  # 台光電
    "2357",  # 華碩
    "2412",  # 中華電
    "6239",  # 力成
    "3034",  # 聯詠
    "6669",  # 緯穎 
    "3017",  # 奇鋐 
    "8046",  # 南電 
    "2356",  # 英業達 
    "2385",  # 群光 
    "8210",  # 勤誠 
    "3026",  # 禾伸堂 
    "8150",  # 南茂
    "3533",  # 嘉澤 
    "3005",  # 神基
    "2449",  # 京元電子 
    "3014",  # 聯陽 
    "6667",  # 信紘科 
]

'''
#交易前查看用戶庫存
print("\nBefore trading:\n")
print(user.Get_User_Stocks())

查看股票資料
stock_id = "6669" 
start_date = "20250515" #起始時間 格式YYYYMMDD
end_date = "20250516" #結束時間 格式YYYYMMDD

個股資訊
print("\nStock Info:\n")
print(Stock_API.Get_Stock_Informations(stock_id,start_date,end_date))

購買股票
stock_id = "6669" 
share = 1 #張數
price = 1890 #價錢
user.Buy_Stock(stock_id,share,price)

售出股票
stock_id = "6669" 
share = 2 #張數
price = 2400 #價錢
user.Sell_Stock(stock_id,share,price)

交易後查看用戶庫存
print("\nAfter trading:\n")
print(user.Get_User_Stocks())
'''



#使用訓練好的模型預測股票漲跌並選出適當股票進入股票池供決策模組使用
def transformer_predict(tech_stocks,rolling_time,today_date,scaler):

    pos_pool=[]
    neg_pool=[]

    #取得每支股票前10天資料
    all_abstract_stocks = []
    for stock_id in tech_stocks:
        stock_data = Stock_API.Get_Stock_Informations(stock_id,rolling_time,today_date)
        #取出資料中開盤價、收盤價、最高價、最低價、成交量、漲跌幅
        selected_features = ['open', 'close', 'high', 'low', 'transaction_volume', 'change']
        abstract_data = [[item[f] for f in selected_features] for item in stock_data]
        
        # 反轉數據順序：從 rolling_time 到 today_date（從舊到新）
        abstract_data = abstract_data[::-1]
        
        all_abstract_stocks.append(abstract_data)
    

    #若股票預測資料不為10筆則退出
    if len(all_abstract_stocks[0]) != 10:
        print("股票資料不正確:")
        print(len(all_abstract_stocks[0]))
        exit()
    #將資料轉換為模型可輸入的格式
    print((len(all_abstract_stocks), len(all_abstract_stocks[0]), len(all_abstract_stocks[0][0])))
    
    #將預測資料進行標準化
    # 對每支股票分別進行標準化
    scaled_stocks = []
    for stock in all_abstract_stocks:
        stock_array = np.array(stock)  # 形狀: (time_steps, features)
        scaled_stock = scaler.transform(stock_array)
        scaled_stocks.append(scaled_stock)
    
    all_abstract_stocks = np.array(scaled_stocks)
    
    #模型預測
    model = StockTransformer(input_dim=6, d_model=128, nhead=4, num_layers=2, dropout=0.1)
    model.load_state_dict(torch.load("./model/stock_transformer.pth"))
    model = model.to(device)  # 將模型移到GPU
    model.eval()


    all_abstract_stocks = torch.tensor(all_abstract_stocks).float()  # shape: (所有股票, 10, 6) 
    all_abstract_stocks = all_abstract_stocks.to(device)  # 將輸入數據移到GPU

    with torch.no_grad():
        #輸出模型預測結果
        predictions = model(all_abstract_stocks)

        #將預測結果由漲跌分類並輸出各自index
        pos_pool = [i for i in range(len(predictions)) if predictions[i] > 0]
        neg_pool = [i for i in range(len(predictions)) if predictions[i] <= 0]

    #回傳漲跌幅由大到小排列的股票index
    return pos_pool,neg_pool;


    # Stock_API.Get_Stock_Informations(stock_id,start_date,end_date)
if __name__ == "__main__":

    #預測漲的股票池
    buy=[] 
    #預測跌的股票池
    sell=[]
    #今天日期
    today_date ='20250529'
    #每次購入10張股票
    buy_share=2
    sell_share=2

    rolling_time = (datetime.strptime(today_date, "%Y%m%d") - timedelta(days=time_window)).strftime("%Y%m%d")

    #載入訓練時的scaler
    scaler = joblib.load(scaler_path)
    buy,sell=transformer_predict(tech_stocks,rolling_time,today_date,scaler)
    #取得所有股票前10天的資料
    print("----------------------------------------------------")
    print("預測漲的股票:")
    print([tech_stocks[i] for i in buy])
    print("預測跌的股票:")  
    print([tech_stocks[i] for i in sell])
    print("----------------------------------------------------")



    # #根據預測結果的股票下單
    # for stock_index in buy:
    #     #先找出該股票當天閉盤價
    #     print("買入:")
    #     print(tech_stocks[stock_index])
    #     the_stock_close_price = Stock_API.Get_Stock_Informations(tech_stocks[stock_index],today_date,today_date)
    #     #購買股票
    #     user.Buy_Stock(tech_stocks[stock_index],buy_share,the_stock_close_price[0]['close'])




    # #根據預測跌的股票若有持有則賣出
    # for stock_index in sell:
    #     #先找出該股票當天閉盤價
    #     the_stock_close_price = Stock_API.Get_Stock_Informations(tech_stocks[stock_index],today_date,today_date)
    #     #若預測跌的股票有持有則賣出
    #     if(any(item['stock_code_id'] == tech_stocks[stock_index] for item in user.Get_User_Stocks())):
    #         print("賣出:")
    #         print(tech_stocks[stock_index])
    #         user.Sell_Stock(tech_stocks[stock_index],sell_share,the_stock_close_price[0]['close'])



    # #購買股票後查看用戶庫存
    # print("\nAfter trading:\n")
    # print( )


