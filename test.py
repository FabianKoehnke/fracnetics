import pandas as pd 
import time 

start = time.perf_counter()
test = pd.read_csv("data/IRIS.csv")

sum = 0
for k in range(10_000):
    for i in range(test.shape[0]):
        sum += test.iloc[i][0]
end = time.perf_counter()
t = end-start
print(f"sum = {sum} done in {t} sek.")