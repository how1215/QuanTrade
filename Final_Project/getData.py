from Stock_API import Stock_API
from datetime import datetime, timedelta
import pandas as pd
import yaml
import os
from tqdm import tqdm
from typing import List, Dict

#用戶登入
user = Stock_API('P76131717' , 'P76131717')
#訓練資料時長
start_date = "20240515"
end_date = "20250523"


def load_yaml_to_dict(file_path: str) -> Dict:
    """
    從 YAML 檔案讀取並轉換為字典。
    :param file_path: YAML 檔案路徑
    :return: 轉換後的字典
    """
    try:
        with open(file_path, 'r') as yaml_file:
            return yaml.load(yaml_file, Loader=yaml.FullLoader)
    except Exception as e:
        print(f"讀取 YAML 檔案時出錯: {e}")
        return {}


def save_dict_to_yaml(data_dict: Dict, file_path: str) -> None:
    """
    將字典儲存為 YAML 格式的檔案。
    :param data_dict: 要儲存的字典
    :param file_path: 儲存的檔案路徑
    """
    try:
        with open(file_path, 'w') as yaml_file:
            yaml.dump(data_dict, yaml_file, default_flow_style=False, allow_unicode=True)
        print(f"字典已成功儲存到 {file_path}")
    except Exception as e:
        print(f"儲存 YAML 檔案時出錯: {e}")


def download_stock_data(stock_code_list: List[str], start_date: str, end_date: str, save_dir: str = './tmp') -> pd.DataFrame:
    """
    下載並儲存股票資料，如果已有資料且時間範圍符合，則直接讀取現有資料
    
    :param stock_code_list: 股票代碼列表
    :param start_date: 開始日期 (YYYYMMDD)
    :param end_date: 結束日期 (YYYYMMDD)
    :param save_dir: 儲存目錄
    :return: 股票資料的DataFrame
    """
    # 檢查是否有現有的資料可用
    if os.path.exists(f'{save_dir}/stock_data.csv') and os.path.exists(f'{save_dir}/save_data_info.yaml'):
        yaml_dict = load_yaml_to_dict(f'{save_dir}/save_data_info.yaml')
        
        yaml_start_date = datetime.strptime(yaml_dict['start_date'], "%Y%m%d")
        yaml_end_date = datetime.strptime(yaml_dict['end_date'], "%Y%m%d")
        
        _start_date = datetime.strptime(start_date, "%Y%m%d") - timedelta(90)
        _end_date = datetime.strptime(end_date, "%Y%m%d")
        
        if _start_date >= yaml_start_date and _end_date <= yaml_end_date:
            print("載入現有股票資訊")
            stock_data = pd.read_csv(f'{save_dir}/stock_data.csv', dtype={"stock_code": str, "date": str})
            stock_data.set_index(["stock_code", "date"], inplace=True)
            stock_data.sort_index(inplace=True)
            return stock_data
    
    print("載入所有股票數據中...")
    all_data = []
    # 提前拿三個月資料以免沒有前幾天資料
    modified_start_date = (datetime.strptime(start_date, "%Y%m%d") - timedelta(90)).strftime("%Y%m%d")
    
    for stock_code in tqdm(stock_code_list, desc="下載股票資料"):
        result = Stock_API.Get_Stock_Informations(stock_code, modified_start_date, end_date)
        
        if not isinstance(result, list):
            print(f"警告: {stock_code} API 回傳數據格式錯誤，跳過")
            continue

        for data in result:
            if not isinstance(data, dict):
                print(f"錯誤: {stock_code} 的資料格式不正確 -> {data}")
                continue

            if "date" not in data or "close" not in data:
                print(f"缺少關鍵欄位: {stock_code} - {data}")
                continue

            # 確保 `stock_code` 也加進 data
            data["stock_code"] = stock_code
            data["date"] = datetime.utcfromtimestamp(data["date"]).strftime('%Y%m%d')
            all_data.append(data)

    if not all_data:
        print("警告: 沒有可用的股票數據，請檢查 API 回應")
        return pd.DataFrame()

    stock_data = pd.DataFrame(all_data)
    stock_data['stock_code'] = stock_data['stock_code'].astype(str)

    # 檢查 DataFrame 是否包含 `stock_code` 和 `date`
    if "stock_code" not in stock_data.columns or "date" not in stock_data.columns:
        print("錯誤: stock_data 缺少 stock_code 或 date 欄位，請檢查數據格式")
        print(stock_data.head())  # Debugging
        return pd.DataFrame()

    stock_data.set_index(["stock_code", "date"], inplace=True)
    stock_data.sort_index(inplace=True)
    print("股票數據載入完成！")
    
    # 儲存資料
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    save_dict_to_yaml({"start_date": modified_start_date, "end_date": end_date}, f"{save_dir}/save_data_info.yaml")
    stock_data.to_csv(f'{save_dir}/stock_data_1.csv')
    
    return stock_data


# 取得所有股票資訊
stock_list = Stock_API.get_all_stock_information()

# 下載股票資料
stock_data = download_stock_data(stock_list, start_date, end_date,save_dir='./tmp')

# 查看資料結構
print("下載的股票資料筆數:", len(stock_data))
print("資料預覽:")
print(stock_data.head())