# How to run the code

## file organization
- auto_test.py : Python script to run fszy.py automaticly. You can test different data size and fusion rate. 

- fzsy.py : Test the performance of TARNet. You need to manually specify the parameters that are passed.
- fzsy_dbl.py : Test the performance of TARNet. The source of the experimental results in the paper, has been specified the fusion ratio and the corresponding data scale.
- fzsy_dbl_cf.py : Test the performance of Causal Forest. The source of the experimental results in the paper, has been specified the fusion ratio and the corresponding data scale

- shiyanfenxi.py : Used to process the output of fzsy, called in auto test.

- huitu.py : The code used to draw the figures in the paper

- util.py : Implementation of causal forest.

- result_cf(tarnet).txt : Results of the experiment.

## simulation experiment

parameters of the fzsy.py

    --n : data size of the test
    --r fusion rate
    --l whether load from file already exists, iff 'y' means load from file already exists
    --t (optional) The label for which training session

one demo
```python
python fzsy.py --n 1000 --r 3 --l y --t 1
```

## Automated testing

run auto_test.py 

    line 4 : for data_size in [{list of data size you want test}]:
    line 6 : for rate in tqdm.tqdm([{fusion rate you want test}]):
    line 7 : num = {the times of test you want try in one setting (n=xxx,r=xxx)}

## Calculate matrics

if you use auto_test, you should first move content of log.txt to result.txt and delete unrelated columns and rows, then replace \tab by ',' to make it as the csv format. Then run the script to get the matrics.

## Get the results in the paper

Run fzsy_dbl_cf.py and fzsy_dbl.py. And calculate the mean and variance of the groups in the resulting log.

##  Drawing the picture

Run huitu.py

##  Check out the experimental results obtained in the paper

Look at result_cf.txt and result_tarnet.txt.

    group information:
        n=xxx,rate=xxx (means data size is xxx and fusion ration is xxx)
        train : xxx test : xxx (means train data set is xxx and test set is xxx)
    n : 1000 , 5000, 10000
    rate : 1, 3, 5
    train : base(biased rct), ours(fused data)
    test : b(biased rct), g(large scale unbiased rct)
        


