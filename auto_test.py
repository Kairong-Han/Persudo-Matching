import subprocess
import tqdm

for data_size in [1000,5000,10000]:
    print(f'data_size:{data_size}')
    for rate in tqdm.tqdm([1,3,5]):
        num = 10
        # 自动化运行其他py文件

        with open('log.txt', 'w') as file:
            pass

        for i in tqdm.trange(num):
            if i == 0:
                result = subprocess.run(
                    ['python', 'fzsy.py','--n', f'{data_size}','--r',f'{rate}','--l', 'n','--t',str(i+1)],  # 将脚本名与参数拼接成命令
                    capture_output=True,
                    text=True
                )
            else:
                result = subprocess.run(
                    ['python', 'fzsy.py','--n', f'{data_size}','--r',f'{rate}','--l', 'y','--t',str(i+1)],  # 将脚本名与参数拼接成命令
                    capture_output=True,
                    text=True
                )


        result = subprocess.run(
                ['python', 'shiyanfenxi.py','--n', f'{data_size}','--r',f'{rate}'],  # 将脚本名与参数拼接成命令
                capture_output=True,
                text=True
            )
