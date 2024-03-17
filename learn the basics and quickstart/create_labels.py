

def write(writer, name, label):
    group = name + "," + str(label) + "\n"
    print(group)
    writer.write(group)

end=".jpg"
comb = ""
label = 1

wr = open("labels.csv","w")
wr.close()

wr = open("labels.csv",'a')
for i in range (0,776):
    name = str(i+1)+"_positive_roi" + end
    write(wr,name, label)

label=0
for i in range (0,776):
    name = str(i+1)+"_negative_roi" + end
    write(wr,name, label)

print("finished")

import pandas as pd
df = pd.read_csv('labels.csv')
print(df.to_string())
