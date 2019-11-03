import sys; sys.path.append('../')
import Black_White_Chess.board as Board
import Black_White_Chess.PlayerSet.Random_Player as RPlayer
import Black_White_Chess.PlayerSet.MiniMax_Player as MPlayer
import Black_White_Chess.game as Game
import Black_White_Chess.PlayerSet.Human_Player as HPlayer
import Black_White_Chess.PlayerSet.AlphaBeta_Player as ABplayer
import Black_White_Chess.PlayerSet.MCTree_Player as MCTPlayer
import Black_White_Chess.PlayerSet.Hybrid_Player as HybridPlayer
import time
import os


def Main():
    # 初始化两个随机玩家
    # rPlayer1 = RPlayer.RandomPlayer('X')
    # rPlayer1 = MPlayer.MiniMaxPlayer('X')
    rPlayer1 = ABplayer.AlphaBetaPlayer('X')
    # rPlayer1 = MCTPlayer.MCTreePlayer('X')
    # rPlayer1 = HybridPlayer.HybridPlayer('X')

    rPlayer2 = HybridPlayer.HybridPlayer('O')

    totalSet = [0, 0, 0]
    diffSet = [0, 0, 0]
    N = 10
    for i in range(N):
        # 初始化游戏
        if i % 2 == 0:
            game = Game.Game(rPlayer1, rPlayer2)
        else:
            game = Game.Game(rPlayer2, rPlayer1)
        # 运行游戏
        result, diff = game.run()
        if result == 'black_win':
            totalSet[i % 2] += 1
            diffSet[i % 2] += diff
        elif result == 'white_win':
            totalSet[1 - i % 2] += 1
            diffSet[1 - i % 2] += diff
        else:
            totalSet[2] += 1
            diffSet[2] += diff

    print('res:', totalSet)
    print('dif:', diffSet)

    print('P1 winRatio:', totalSet[0] / N)
    print('P1 difRatio:', diffSet[0] / N)
    print('P2 winRatio:', totalSet[1] / N)
    print('P2 difRatio:', diffSet[1] / N)



if __name__ == '__main__':
    Main()
    # print([[] for _ in range(8)])
