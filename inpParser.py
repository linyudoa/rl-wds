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
                print("stop printing, ", count, "objects in total for filed", indicateStr)
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
        self.aField = mp

    def summarizeaField(self, fieldName):
        if (fieldName == "[PATTERNS]"):
            field = self.patterns
        elif (fieldName == "[DEMANDS]"):
            field = self.demands
        else:
            print("invalid field name, should either be [PATTERNS] or [DEMANDS]")
            return
        if (len(self.aField) == 0):
            print("Field empty")
            return
        for key in self.aField.keys():
            print("Key: ", key)
            print(len(self.aField[key]))

pathToWds = "water_networks/QDMaster1031_master.inp"
parser = MyParser(pathToWds)
parser.readField("[PATTERNS]")
parser.readField("[DEMANDS]")
parser.summarizeaField("[DEMANDS]")
