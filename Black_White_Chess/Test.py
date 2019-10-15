import sys; sys.path.append('../')
import Black_White_Chess.board as Board
import Black_White_Chess.game as Game
import Black_White_Chess.RandomPlayer as RPlayer
import time
import os


# 执行操作
def TakeAction(player, board):
    action = player.get_move(board)
    if action is not None:
        player.move(board, action)
        return False
    else:
        return True


def Main():
    # 棋盘初始化
    board = Board.Board()

    # 初始化两个随机玩家
    rPlayer1 = RPlayer.RandomPlayer('X')
    rPlayer2 = RPlayer.RandomPlayer('O')

    xflag = True
    endFlag = False
    while endFlag is False:
        if xflag is True:
            endFlag = TakeAction(rPlayer1, board)
        else:
            endFlag = TakeAction(rPlayer2, board)
        xflag = not xflag
        board.display()

        time.sleep(3)
        os.system("clear")



if __name__ == '__main__':
    Main()