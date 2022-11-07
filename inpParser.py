class MyParser():
    def __init__(self,
            pathToInp):
        self.pathToInp = pathToInp
        self.lines = []
        self.loadFile()
        self.patterns = {}
        self.demands = {}

    def loadFile(self):
        fileHandler = open(self.pathToInp, "r")
        self.lines = fileHandler.readlines()
    
    def readField(self, indicateStr):
        vldFlag = False
        mp = {}
        count = 0
        for line in self.lines:
            lineItems = line.strip().rstrip(';').split()
            if (vldFlag == True and len(lineItems) == 0):
                vldFlag = False
                break
            elif (len(lineItems) == 0):
                continue
            if (vldFlag):
                key = lineItems[0]
                vals = lineItems[1:]
                for val in vals:
                    if (key in mp.keys()):
                        mp[key].append(val)
                    else:
                        mp[key] = [val]
                        count += 1
            if (lineItems[0] == indicateStr):
                vldFlag = True
            if (indicateStr == "[PATTERNS]"):
                self.patterns = mp
            elif (indicateStr == "[DEMANDS]"):
                self.demands = mp
            else:
                print("invalid field name, should either be [PATTERNS] or [DEMANDS]")

    def summarizeField(self, fieldName):
        if (fieldName == "[PATTERNS]"):
            field = self.patterns
        elif (fieldName == "[DEMANDS]"):
            field = self.demands
        else:
            print("invalid field name, should either be [PATTERNS] or [DEMANDS]")
            return
        if (len(field) == 0):
            print("Field empty")
            return
        for key in field.keys():
            print("Key: ", key)
            print(len(field[key]))
    


pathToWds = "water_networks/QDMaster1031_master.inp"
parser = MyParser(pathToWds)
parser.readField("[PATTERNS]")
parser.readField("[DEMANDS]")
parser.summarizeField("[PATTERNS]")
