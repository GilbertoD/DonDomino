from tqdm import trange, tqdm

# Package for play games in https://dominoes.playdrift.com/
import play
import numpy as np

# Process the acquired information 
import processing

import pickle

# nPlays = 1000
# for _ in trange(nPlays):
#     try : play.play()
#     except : continue

# STATES, ACTIONS = [],[]
# gamesPath = processing.gamesPath()
# fileList = processing.fileList()
#
# fileName = 'DATA.h5'
# DATA = {'s':[], 'a':[]}
# with open(fileName, "wb") as f:
#     pickle.dump(DATA, f)
#
# for f in tqdm( fileList ) :
#     try:
#         with open(fileName, 'rb') as r:
#             DATA = pickle.load(r)
#
#         strFile = gamesPath + '/' + f
#         s,a = processing.process(strFile)
#
#         DATA['s'].extend( s )
#         DATA['a'].extend( a )
#
#         with open(fileName, "wb") as f:
#             pickle.dump(DATA, f)
#
#     except : continue
#
# print(f'# States {len(DATA["s"])} # Actions {len(DATA["a"])}')


fileName = './games/DATA.h5'
with open(fileName, 'rb') as r:
    DATA = pickle.load(r)

STATES, ACTIONS = DATA['s'], DATA['a']
print(f'# States {len(STATES)} # Actions {len(ACTIONS)}')
