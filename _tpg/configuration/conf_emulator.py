from _tpg.emulator import Emulator

class ConfEmulator:
    def init_def(self, team, functionsDict, num=1, actVars=None):
        self.team = team
        self.functionsDict = functionsDict
        self.emuNum = num
        self.actVars = actVars
    
    def step_def(self, act, prestate): pass

    def reconfirmation_def(self, states): pass

    def saveToFile(self, filename): pass