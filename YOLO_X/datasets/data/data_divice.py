import numpy as np

np.random.seed(10101)
list_samples = np.random.permutation(range(1, 9963))
num_test = int(9963 * 0.15)
num_val=int(9963*0.05)
num_train = 9963 - num_val-num_test
train_idx_list = list_samples[:num_train]
val_idx_list = list_samples[num_train:(num_train+num_val)]
test_idx_list=list_samples[(num_train+num_val):]
with open("train.txt","w")as f:
    for x in train_idx_list:
        f.write(str(x)+"\n")

with open("val.txt","w")as f:
    for x in val_idx_list:
        f.write(str(x)+"\n")

with open("test.txt","w")as f:
    for x in test_idx_list:
        f.write(str(x)+"\n")
