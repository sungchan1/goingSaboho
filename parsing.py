

import matplotlib.pyplot as plt


fp = open('message_ADSNY.txt', 'r')
lines = fp.readlines()

precision_list = list()
recall_list = list()
f1_list = list()
x_range = range(1,99)
for line in lines:
    item = line.split(" ")
    # prinst(item)
    # print(item.index('val_precision:')+1)
    if 'val_recall:' in item:
        precision_list.append(float(item[item.index("val_recall:")+1])) 
        recall_list.append(float(item[item.index("val_precision:")+1]))
        f1_list.append(2 * float(recall_list[-1]) * float(precision_list[-1]) / (float(recall_list[-1]) + float(precision_list[-1])))


for i in x_range :
    print('Epoch ', i , 'precision : ',precision_list[i-1], ' recall : ', recall_list[i-1] ,' f1 : ', f1_list[i-1])
plt.figure(figsize=(10,10))
plt.plot(x_range, precision_list, label='precision')
plt.plot(x_range, recall_list, label='recall')
plt.plot(x_range, f1_list, label='f1_score')
plt.xlabel('Epochs')
plt.ylabel('Score')
# 그림에 선 표시
# plt.grid(True)
# 범례 표시: best - 자동으로 최적의 위치에
plt.legend(loc="best")
plt.show()
plt.savefig('recall_precision_ADSNY.png')
fp.close()