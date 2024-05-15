import pandas as pd  # 一定记得改通路条数！！
import os
import re

from numpy.core.defchararray import isdigit

data = pd.read_csv("15-ZL-000007-Ca1.csv")
os.mkdir(r"D:\users\p30057372\some_script\2024-5-15\15-ZL-000007-Ca1-7条通路")
header = pd.read_excel(r"D:\users\p30057372\some_script\2024-5-15\s2.xlsx").keys().tolist()

rstr = r"[\/\\\:\*\?\"\<\>\|]"
for i in range(len(header)):
    header[i] = re.sub(rstr, " ", header[i])

for j in range(7):
    file = r"D:\users\p30057372\some_script\2024-5-15\15-ZL-000007-Ca1-7条通路\{name}.csv".format(name=header[j])
    source = pd.read_excel(r"D:\users\p30057372\some_script\2024-5-15\s2.xlsx").iloc[:, j].values.tolist()
    source = [x for x in source if not pd.isnull(x)]
    for i in source:
        result = data[data.iloc[:, 0].apply(lambda x: x.lower() if not isdigit(x) else "unknow") == i.lower()]
        if result.empty:
            print("查不到基因{name}的相关数据".format(name=i))
        if not os.path.exists(file):
            result.to_csv(file, header=True, mode="w", index=False, encoding="utf_8_sig")
        else:
            result.to_csv(file, header=False, mode="a", index=False, encoding="utf_8_sig")
