inpLines = []
pathToInp = "results\logspeed.txt"
fileHandler = open(pathToInp, "r", encoding='latin1')
inpLines = fileHandler.readlines()
mp = {}
index = 0
for line in inpLines:
    line = line.strip()
    lineItems = line.strip().split()
    vals = lineItems[1:]
    for val in vals:
        if (index in mp.keys()):
            mp[index].append(val)
        else:
            mp[index] = [val]
    index += 1

for index in range(1152, 1440):
    print(mp[index][0])