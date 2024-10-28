# How to run the code

## 1. simulation experiment

parameters of the .py

    --n : data size of the test
    --r fusion rate
    --l whether load from file already exists, iff 'y' means load from file already exists
    --t (optional) The label for which training session

one demo
```python
python --n 1000 --r 3 --l y --t 1
```

## Automated testing

run auto_test.py 

    line 4 : for data_size in [{list of data size you want test}]:
    line 6 : for rate in tqdm.tqdm([{fusion rate you want test}]):
    line 7 : num = {the times of test you want try in one setting (n=xxx,r=xxx)}

## Calculate matrics

you should first move content of log.txt to result.txt and delete unrelated columns and rows, then replace \tab by ',' to make it as the csv format. Then run the script to get the matrics.

