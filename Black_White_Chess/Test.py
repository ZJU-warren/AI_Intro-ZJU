import sys; sys.path.append('../')
import Black_White_Chess.board as Board
import Black_White_Chess.PlayerSet.Random_Player as RPlayer
import Black_White_Chess.PlayerSet.MiniMax_Player as MPlayer
import Black_White_Chess.game as Game
import Black_White_Chess.PlayerSet.Human_Player as HPlayer
import Black_White_Chess.PlayerSet.AlphaBeta_Player as ABplayer
import Black_White_Chess.PlayerSet.MCTree_Player as MCTPlayer
import time
import os


def Main():
    start = time.time()

    # 初始化两个随机玩家
    # rPlayer1 = MPlayer.MiniMaxPlayer('X')
    # rPlayer1 = ABplayer.AlphaBetaPlayer('X')
    rPlayer1 = MCTPlayer.MCTreePlayer('X')          # !!! some bug
    # rPlayer1 = RPlayer.RandomPlayer('X')

    # rPlayer2 = RPlayer.RandomPlayer('O')
    rPlayer2 = ABplayer.AlphaBetaPlayer('O')

    totalSet = [0, 0, 0]
    diffSet = [0, 0, 0]
    N = 10
    for i in range(N):
        # 初始化游戏
        game = Game.Game(rPlayer1, rPlayer2)

        # 运行游戏
        result, diff = game.run()
        if result == 'black_win':
            totalSet[0] += 1
            diffSet[0] += diff
        elif result == 'white_win':
            totalSet[1] += 1
            diffSet[1] += diff
        else:
            totalSet[2] += 1
            diffSet[2] += diff

    print('res:', totalSet)
    print('dif:', diffSet)

    print('winRatio:', totalSet[0]/N)
    print('difRatio:', diffSet[0]/N)

    end = time.time()
    print('time cost:', end - start)


if __name__ == '__main__':
    Main()
    # print([[] for _ in range(8)])
