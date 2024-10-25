import csv
import numpy as np
import os


LOG_DIR = "./Design_C_log"

data = []
with open(os.path.join(LOG_DIR, "eval_env_monitor.log.monitor.csv"), 'r') as file:
    reader = csv.reader(file)
    _ = next(reader); _ = next(reader)
    
    for j in range(5):
        subdata = []
        for i in range(10):
            subdata.append(float(next(reader)[0]))

        data.append(np.mean(subdata))


print("Mean rewards from runs - ", data)
print("Overall mean - ", np.mean(data))
print("Std over runs - ", np.std(data))
