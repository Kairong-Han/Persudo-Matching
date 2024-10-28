import pandas as pd

import argparse

# 创建解析器
parser = argparse.ArgumentParser(description='处理命令行参数示例')
parser.add_argument('--n', type=int, help='n个数据点', required=True)
parser.add_argument('--r', type=int, help='融合比例', required=True)
# parser.add_argument('--verbose', action='store_true', help='输出详细信息')
args = parser.parse_args()

n = args.n
rate = args.r

with open('result.txt','a+') as f:
    f.write(f'n={n},rate=1:{rate}\n')
    column_names = ['qini', 'mape', 'copc']
    df = pd.read_csv("log.txt",header=None,names=column_names)
    # 读取 CSV 文件，不将第一行作为列名
    print("bias RCT")
    f.write(f'bias RCT\n')
    df_1_multiples = df.iloc[::6]
    print(df_1_multiples.describe().iloc[1:3])
    f.write(f'{df_1_multiples.describe().iloc[1:3]}\n')
    # 取出行索引为 2 的倍数行
    # df_1_multiples = df.iloc[1::6]
    # print(df_1_multiples.describe())

    # 取出行索引为 3 的倍数行
    df_1_multiples = df.iloc[2::6]
    print(df_1_multiples.describe().iloc[1:3])
    f.write(f'{df_1_multiples.describe().iloc[1:3]}\n')

    # 取出行索引为 3 的倍数行
    f.write(f'golden RCT\n')
    print("golden RCT")
    df_1_multiples = df.iloc[3::6]
    print(df_1_multiples.describe().iloc[1:3])
    f.write(f'{df_1_multiples.describe().iloc[1:3]}\n')

    # 取出行索引为 2 的倍数行
    # df_1_multiples = df.iloc[4::6]
    # print(df_1_multiples.describe())

    # 取出行索引为 3 的倍数行
    df_1_multiples = df.iloc[5::6]
    print(df_1_multiples.describe().iloc[1:3])
    f.write(f'{df_1_multiples.describe().iloc[1:3]}\n')

