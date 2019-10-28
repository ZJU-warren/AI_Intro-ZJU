class Node:
    def __init__(self, status, father, action, color):
        self.id = self.Encode(status)
        self.child = []
        self.Q = 0
        self.N = 0
        self.father = father
        self.action = action
        self.color = color          # 完成本步的颜色

    def s2v(self, s):
        if s == 'X': return 1
        elif s == 'O': return 2
        if s == '.': return 0

    def v2s(self, v):
        if v == 1: return 'X'
        elif v == 2: return 'O'
        if v == 0: return '.'

    def Encode(self, status):
        id = 0
        for i in range(8):
            for j in range(8):
                bitValue = self.s2v(status[i][j])
                id = id * 3 + bitValue
        return id

    def Decode(self):
        value = self.id
        status = [['.' for _ in range(8)] for _ in range(8)]
        for i in range(7, -1, -1):
            for j in range(7, -1, -1):
                status[i][j] = self.v2s(value % 3)
                value = value // 3
        return status
