import sys; sys.path.append('../')
import random
from Black_White_Chess.PlayerSet.player import Player
# import copy
import time


class AlphaBetaPlayer(Player):
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

    # min
    def min_value(self, board, alpha, beta):
        action_list = self.OptList(board, self.flipColor())
        if len(action_list) == 0:
            # print('-----------------------------')
            return None, board.count(self.color)

        bestAction = None
        minScore = 1000
        for each in action_list:
            # min最大值
            flipSet = board._move(each, self.flipColor())
            action, score = self.max_value(board, alpha, beta)
            board.backpropagation(each, flipSet, self.flipColor())

            # 更新最优解
            if minScore > score:
                minScore = score
                bestAction = each

            # 减枝
            if minScore <= alpha:
                break
            beta = min(beta, minScore)

        return bestAction, minScore

    # max
    def max_value(self, board, alpha, beta):
        action_list = self.OptList(board, self.color)
        if len(action_list) == 0:
            # print('+++++++++++++++++++++++++++++')
            return None, board.count(self.color)

        bestAction = None
        maxScore = -1000
        for each in action_list:
            # min最大值
            flipSet = board._move(each, self.color)
            action, score = self.min_value(board, alpha, beta)
            board.backpropagation(each, flipSet, self.color)

            # 更新最优解
            if maxScore < score:
                maxScore = score
                bestAction = each

            # 减枝
            if maxScore >= beta:
                break
            alpha = max(alpha, maxScore)

        return bestAction, maxScore

    # alpha_beta搜索
    def alpha_beta_decision(self, board):
        bestAction, maxScore = self.max_value(board, -1000, 1000)
        return bestAction

    def random_choice(self, board):
        # 用 list() 方法获取所有合法落子位置坐标列表
        action_list = self.OptList(board, self.color)
        # 如果 action_list 为空，则返回 None,否则从中选取一个随机元素，即合法的落子坐标
        if len(action_list) == 0:
            return None
        else:
            return action_list[0]

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
        print("请等一会，对方 {}-{} 正在思考中...".format(player_name, self.color))
        sum = board.count(self.flipColor()) + board.count(self.color)
        print('{}', sum)
        if sum < 54:
            action = self.random_choice(board)
        else:
            action = self.alpha_beta_decision(board)
        # while True:
        #     x = input('is END?')
        #     if x == 'END':
        #         break
        return action
