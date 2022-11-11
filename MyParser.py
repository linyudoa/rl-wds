from functools import reduce

class MyParser():
    """Resolve WDS state from input file"""
    def __init__(self,
            pathToInp):
        self.pathToInp = pathToInp
        self.lines = []
        self.loadFile()
        self.patterns = {}
        self.demands = {}
        self.pumps = {}

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
            elif (indicateStr == "[PUMPS]"):
                self.pumps = mp
            else:
                print("invalid field name, should either be [PATTERNS] or [DEMANDS]")

    def summarizeField(self, fieldName):
        if (fieldName == "[PATTERNS]"):
            field = self.patterns
        elif (fieldName == "[DEMANDS]"):
            field = self.demands
        elif (fieldName == "[PUMPS]"):
            field = self.pumps
        else:
            print("invalid field name, should either be [PATTERNS] or [DEMANDS]")
            return
        for key in field.keys():
            print("Key: ", key)
            print(field[key][-1])
        if (len(field) == 0):
            print("Field empty")
            return
    
    def demandSnapshot(self, i : int):
        """Create nodal demand of timeStamp i"""
        mp = {}
        # print("Start to calc demands:==============================================================")
        for junc in self.demands.keys():
            demandIndex = 0
            patternIndex = 1
            if (junc not in mp.keys()):
                mp[junc] = 0
            while demandIndex < len(self.demands[junc]):
                patternId = self.demands[junc][patternIndex]
                patternFactorPos = i if len(self.patterns[patternId]) == 288 else i * 5
                if (patternId not in self.patterns.keys()):
                    print("Pattern illegal, should be in filed [PATTERNS]")
                    return 
                mp[junc] += float(self.demands[junc][demandIndex]) * float(self.patterns[patternId][patternFactorPos])
                demandIndex += 2
                patternIndex += 2
        print("total demand of timestamp ", i +  1, "is: ", reduce(lambda x, y : x + y, mp.values()))
        return mp
        
    def pumpSpeedSnapshot(self, i : int):
        """Snapshot of pump speed in timeStamp i"""
        mp = {}
        # print("Start to calc pump speed:==============================================================")
        for pump in self.pumps.keys():
            if (pump not in mp.keys()):
                mp[pump] = 0
            patternId = self.pumps[pump][-1]
            if (patternId not in self.patterns.keys()):
                print("Pattern illegal, should be in filed [PATTERNS]")
                return 
            patternFactorPos = i if len(self.patterns[patternId]) == 288 else i * 5
            mp[pump] =float(self.patterns[patternId][patternFactorPos])
        print("total pump speed of timestamp ", i +  1, "is: ", reduce(lambda x, y : x + y, mp.values()))
        return mp

# test code
# pathToWds = "water_networks/QDMaster1031_master.inp"
# parser = MyParser(pathToWds)
# parser.readField("[PATTERNS]")
# parser.readField("[DEMANDS]")
# parser.readField("[PUMPS]")
# parser.summarizeField("[PUMPS]")
# for i in range(288):
#     parser.pumpSpeedSnapshot(i)