import sys; sys.path.append('../')
import random
from Black_White_Chess.PlayerSet.player import Player
# import copy
import time
from Black_White_Chess.PlayerSet.Node import *
import Black_White_Chess.board as Board
import math


class MCTreePlayer(Player):
    """
    随机玩家, 随机返回一个合法落子位置
    """

    def __init__(self, color):
        """
        继承基类玩家，玩家初始化
        :param color: 下棋方，'X' - 黑棋，'O' - 白棋
        """
        super().__init__(color)

    # other color
    def flipColor(self):
        return 'X' if self.color == 'O' else 'O'

    def OptList(self, board, color):
        orglist = list(board.get_legal_actions(color))
        goodCol = ['A', 'H']
        badCol = ['B', 'G']
        goodRow = ['1', '8']
        badRow = ['2', '7']

        good_list = []
        bad_list = []
        common_list = []
        for each in orglist:
            if each[0] in goodCol or each[1] in goodRow:
                good_list.append(each)
            elif each[0] in badCol or each[1] in badRow:
                bad_list.append(each)
            else:
                common_list.append(each)
        common_list.extend(bad_list)
        good_list.extend(common_list)

        # print(good_list)
        # time.sleep(2)
        return good_list

    def Expand(self, v, board, action_list, nodeColor):
        for each in action_list:
            flipSet = board._move(each, nodeColor)
            node = Node(board._board, v, each, nodeColor)
            board.backpropagation(each, flipSet, nodeColor)

            flag = True
            for child in v.child:
                if node.id == child.id:
                    flag = False
                    break

            if flag:
                v.child.append(node)
                return node

        print('some wrong here:', 0/0)
        return None

    def BestChild(self, node, c=2.4):
        maxV = -1000
        maxObj = None
        for each in node.child:
            v = each.Q / each.N + c * math.sqrt(2 * math.log(node.N) / each.N)
            if maxV < v:
                maxV = v
                maxObj = each
        return maxObj

    def TreePolicy(self, v):
        colorSet = [self.color, self.flipColor()]
        board = Board.Board()
        board._board = v.Decode()
        deep = 0
        action_list = self.OptList(board, colorSet[deep])
        while len(action_list) != 0:                                            # when v is not terminal
            if len(action_list) > len(v.child):                                 # v not fully expanded
                return self.Expand(v, board, action_list, colorSet[deep])
            else:
                v = self.BestChild(v)
                deep = 1 - deep
                board._board = v.Decode()
                action_list = self.OptList(board, colorSet[deep])
        return v

    def DefaultPolicy(self, node):
        board = Board.Board()
        colorSet = [self.color, self.flipColor()]
        deep = 1 if node.color == self.color else 0                     # 下一步棋颜色

        board._board = node.Decode()
        action_list = list(board.get_legal_actions(colorSet[deep]))

        while len(action_list) > 0:
            action = random.choice(action_list)
            # 转移
            board._move(action, colorSet[deep])
            # node = Node(board._board, node, action, colorSet[deep])
            # board.backpropagation(action, flipSet, colorSet[deep])
            action_list = list(board.get_legal_actions(colorSet[deep]))

        # board._board = node.Decode()
        return board.count(self.color)

    def BackUp(self, node, reward):
        while node is not None:
            node.N += 1
            node.Q += reward
            node = node.father

    # MCTree搜索
    def UCTSearch(self, board):
        root = Node(board._board, None, None, self.flipColor())
        for i in range(30):                        # 枚举1000次
            node = self.TreePolicy(root)
            reward = self.DefaultPolicy(node)
            self.BackUp(node, reward)
        bestAction = self.BestChild(root, 0).action
        return bestAction


    def get_move(self, board):
        """
        根据当前棋盘状态获取最佳落子位置
        :param board: 棋盘
        :return: action 最佳落子位置, e.g. 'A1'
        """
        if self.color == 'X':
            player_name = '黑棋'
        else:
            player_name = '白棋'
        # print("请等一会，对方 {}-{} 正在思考中...".format(player_name, self.color))
        sum = board.count(self.flipColor()) + board.count(self.color)
        # print('{}', sum)
        action_list = self.OptList(board, self.color)
        if len(action_list) == 0:
            return None
        # if sum < 40:
        #     action = action_list[0]
        # else:
        #     action = self.UCTSearch(board)
        action = self.UCTSearch(board)
        # print('_________________________________________________', action)
        # while True:
        #     x = input('is END?')
        #     if x == 'END':
        #         break
        return action
