import os
import glob
import pickle
from copy import deepcopy as dpc

DEBUG = False

gamesPath = lambda : os.path.dirname( os.path.abspath(__file__) )+"/games"

def fileList( ):
    os.chdir( gamesPath() )
    return glob.glob("*.h5")
    # strFile = gamesPath + '/' + f


## STATE CLASS
class playerState:
    IDX = [ [ 0, 1, 2, 3, 4, 5, 6],
            [ 1, 7, 8, 9,10,11,12],
            [ 2, 8,13,14,15,16,17],
            [ 3, 9,14,18,19,20,21],
            [ 4,10,15,19,22,23,24],
            [ 5,11,16,20,23,25,26],
            [ 6,12,17,21,24,26,27] ]

    def __init__( self, id, bones ):

        ZEROS = lambda : [0]*28
        ROTATE = lambda a,n : a[n:] + a[:n]
        self.myIdx = ROTATE( [0,1,2,3], id )    

        self.myBones =  ZEROS()
        for i,j in bones : self.myBones[ self.IDX[i][j] ] = 1            

        self.BOARD = ZEROS()
        self.PASS = [ ZEROS() for _ in range(4) ]

        self.PLAYS = [0]*7

    def update( self, player, action, board ):
        self.PLAYS = [0]*7
        a,_ = board[0]
        _,b = board[-1]
        self.PLAYS[ a ] = 1
        self.PLAYS[ b ] = 1

        if action != 'x' :
            i, j = action 
            self.BOARD[ self.IDX[i][j] ] = 1
        else :
            idxTemp = self.myIdx[ player ]
            for i in self.IDX[a] : self.PASS[ idxTemp ][ i ] = 1
            for i in self.IDX[b] : self.PASS[ idxTemp ][ i ] = 1

    ## State Definition
    # Bones that Player +1 can have (1x28)
    # Bones that Player +2 can have (1x28)
    # Bones that player +3 can have (1x28)
    # Possible plays (1x7)
    def state( self ):
        bones1 = [ int( not ( m or b or p ) ) for m,b,p in zip( self.myBones, self.BOARD, self.PASS[1] ) ]
        bones2 = [ int( not ( m or b or p ) ) for m,b,p in zip( self.myBones, self.BOARD, self.PASS[2] ) ]
        bones3 = [ int( not ( m or b or p ) ) for m,b,p in zip( self.myBones, self.BOARD, self.PASS[3] ) ]
        
        return [ bones1, bones2, bones3, self.PLAYS ]

def process(strFile):
    DATA = None
    with open(strFile, 'rb') as r:
        DATA = pickle.load(r)

    myHand, BOARDS, BONES_NUMBER, final = tuple(DATA)

    if DEBUG :
        for b, n in zip(BOARDS, BONES_NUMBER): print(f'{b} | {n}')

    if DEBUG : print()

    ### POST-PROCESSING
    ## Hands Recontructions
    HANDS = dpc(final)

    players = []
    actions = []

    tempCant = [7, 7, 7, 7]
    tempBoard = []
    for b, n in zip(BOARDS, BONES_NUMBER):
        delta = [a - b for a, b in zip(tempCant, n)]
        plays = [idx for idx, val in enumerate(delta) if val != 0]

        idxPlay = plays[-1]

        if len(players) > 0 and idxPlay != (players[-1] + 1) % 4:
            tempPlayer = (players[-1] + 1) % 4
            while tempPlayer != idxPlay:
                players.append(tempPlayer)
                actions.append('x')

                tempPlayer = (tempPlayer + 1) % 4

        bone = list(set(b) - set(tempBoard))
        bone = bone[-1]
        if bone == b[0]: bone = bone[::-1]

        actions.append(bone)

        HANDS[idxPlay].append(bone)
        players.append(idxPlay)

        tempCant = dpc(n)
        tempBoard = dpc(b)

    # Get Actions
    if DEBUG :
        print(myHand)
        print()
        for hand in HANDS: print(hand)
        print()
        print([])

    ALL_BOARDS = [[]]
    idx = -1
    for p, a in zip(players, actions):
        if DEBUG : print(f'Player {p:d} played {a}')
        if a != 'x': idx += 1
        if DEBUG : print(BOARDS[idx])
        ALL_BOARDS.append(BOARDS[idx])

    # Re-play the game (Get States)
    if DEBUG : print()
    playerStates = [playerState(i, hand) for i, hand in enumerate(HANDS)]

    STATES = []
    ACTIONS = []
    ZEROS = lambda: [[0] * 7 for _ in range(7)]

    for p, a, b in zip(players, actions, ALL_BOARDS[1:]):
        # Get state of player p
        state = playerStates[p].state()

        # Get action of player p
        aTemp = ZEROS()
        if a != 'x':
            i, j = a
            aTemp[i][j] = 1

        # Update all player's states
        for s in playerStates: s.update(p, a, b)

        # Save state and action if player doesn't pass
        if a != 'x' or p == 0:
            STATES.append(state)
            ACTIONS.append(aTemp)

    if DEBUG :
        print(f'# States: {len(STATES)} | # Actions: {len(ACTIONS)}')
        for s, a in zip(STATES, ACTIONS): print(f'{s} ---> {a}\n')

    return STATES,ACTIONS