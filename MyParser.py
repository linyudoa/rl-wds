from functools import reduce

class MyParser():
    """Resolve WDS state from input file"""
    def __init__(self,
            pathToInp, 
            pathToTankSeries = None):
        self.pathToInp = pathToInp
        self.pathToTankSeries = pathToTankSeries
        self.inpLines = []
        self.tankLines = []
        self.tankLevels = {}
        self.patterns = {}
        self.junctions = {}
        self.demands = {}
        self.pumps = {}
        self.initialize()

    def initialize(self):
        fileHandler = open(self.pathToInp, "r", encoding='latin1')
        self.inpLines = fileHandler.readlines()
        fileHandler.close()
        self.readWdsField("[PATTERNS]")
        self.readWdsField("[PUMPS]")
        self.readWdsField("[DEMANDS]")
        self.readWdsField("[JUNCTIONS]")
        if (self.pathToTankSeries):
            fileHandler = open(self.pathToTankSeries, "r", encoding='latin1')
            fileHandler.readline()
            self.tankLines = fileHandler.readlines()
            fileHandler.close()
            self.readTankSeries()

    def readTankSeries(self):
        count = 0
        mp = {}
        for line in self.tankLines:
            lineItems = line.strip().split()
            if (len(lineItems) == 0): continue
            key = count
            vals = lineItems[1:]
            mp[key] = list(map(float, vals))
            count += 1
        self.tankLevels = mp

    def readWdsField(self, indicateStr):
        vldFlag = False
        mp = {}
        for line in self.inpLines:
            line = line.strip().rstrip(';')
            if (line.rfind(';') > -1): 
                pos = line.rfind(';')
                line = line[:pos]
            lineItems = line.strip().split()
            if (len(lineItems) == 0): continue
            if (vldFlag == True and (len(lineItems) == 0 or lineItems[0][0] == '[')):
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
            if (lineItems[0] == indicateStr):
                vldFlag = True
        if (indicateStr == "[PATTERNS]"):
            self.patterns = mp
        elif (indicateStr == "[DEMANDS]"):
            self.demands = mp
        elif (indicateStr == "[PUMPS]"):
            self.pumps = mp        
        elif (indicateStr == "[JUNCTIONS]"):
            self.junctions = mp
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
            print(len(field[key]))
        if (len(field) == 0):
            print("Field empty")
            return
    
    def demandSnapshot(self, i : int):
        mp = {}
        self.fill_demands_from_demands(i, mp)
        self.fill_demands_from_junctions(i, mp)
        return mp

    def tankLevelSnapshot(self, i : int):
        return self.tankLevels[i]

    def fill_demands_from_demands(self, i, mp : dict):
        """Create nodal demand of timeStamp i"""
        for junc in self.demands.keys():
            demandIndex = 0
            patternIndex = 1
            mp[junc] = 0
            while demandIndex < len(self.demands[junc]):
                patternId = self.demands[junc][patternIndex]
                patternFactorPos = i % 288 if len(self.patterns[patternId]) == 288 else i 
                if (patternId not in self.patterns.keys()):
                    print("Pattern illegal, should be in filed [PATTERNS]")
                    return 
                mp[junc] += float(self.demands[junc][demandIndex]) * float(self.patterns[patternId][patternFactorPos])
                demandIndex += 2
                patternIndex += 2
    
    def fill_demands_from_junctions(self, i : int, mp : dict):
        """Create nodal demand of timeStamp i, from [juntions] field"""
        for junc in self.junctions.keys():
            demandIndex = 1
            patternIndex = 2
            if (len(self.junctions[junc]) < 3): continue
            if (junc not in mp.keys()):
                mp[junc] = 0
            else:
                continue
            patternId = self.junctions[junc][patternIndex]
            patternFactorPos = i % 288 if len(self.patterns[patternId]) == 288 else i 
            if (patternId not in self.patterns.keys()):
                print("Pattern illegal, should be in filed [PATTERNS]")
                return 
            mp[junc] += float(self.junctions[junc][demandIndex]) * float(self.patterns[patternId][patternFactorPos])

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
            patternFactorPos = i % 288 if len(self.patterns[patternId]) == 288 else i 
            mp[pump] = float(self.patterns[patternId][patternFactorPos])
        # print("total pump speed of timestamp ", i +  1, "is: ", reduce(lambda x, y : x + y, mp.values()))
        return mp

## test code
pathToWds = "water_networks/QDMaster_master.inp"
pathToTankLevel = "water_networks/QDMaster_master_tank_level.txt"
parser = MyParser(pathToWds, pathToTankLevel)
parser.demandSnapshot(0)
