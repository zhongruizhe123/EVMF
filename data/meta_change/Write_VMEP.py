# -*- coding:utf-8 -*-
# -*- encoding: utf-8 -*-
import json
import csv
from io import StringIO
# from urllib import urlopen

# 按行元组参数写入

def writerCsv1():
    f = open("data.csv", "w")
    writer = csv.writer(f)
    for i in range(100):
        writer.writerow((i+1, i+2, i+3))
    f.close()

# 按行字典参数写入
def writerCsv2():
    f = open("data.csv", "w")
    writer = csv.DictWriter(f, ["name", "age"])
    for i in range(100):
        dct = {"name":i+1, "age": i+2}
        writer.writerow(dct)
    f.close()
    print("写入成功！")
def readerCsv3():
    readjson = json.load(open('gen_res_val2.json', 'r', encoding="utf-8"))
    with open('val_2_meta.csv','r', encoding='UTF-8') as f:
        tsvreader = csv.reader(f, delimiter='\t')
        flag = 1
        for line in tsvreader:
            # if flag ==2:
            #     print(line)
            # flag+=1
            if flag == 1:
                line.append('imgs')

            else:
                idx = line[8]
                imgscaption = readjson[idx]['gen'][0]
                line.append(imgscaption)
            flag+=1
            print(line)
            with open(r'val_2_meta_nosub.csv', 'a', encoding='UTF-8', newline='') as f:
                tsv_w = csv.writer(f, delimiter='\t')
                tsv_w.writerow(line)

readerCsv3()

