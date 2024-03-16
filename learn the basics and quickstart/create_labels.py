
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
    name = str(i+1)+end
    write(wr,name, label)

print("finished")
