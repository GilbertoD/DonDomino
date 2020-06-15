import time
import datetime
import re
from copy import deepcopy as dpc
from selenium import webdriver
from selenium.webdriver.support.ui import Select
from selenium.webdriver.common.action_chains import ActionChains
from selenium.common.exceptions import StaleElementReferenceException
import pickle
import datetime

DEBUG = False

BONES = [ [(6,6), (6,5), (6,4), (6,3), (6,2), (6,1), (6,0)],
          [(5,5), (5,4), (5,3), (5,2), (5,1), (5,0), (4,4)],
          [(4,3), (4,2), (4,1), (4,0), (3,3), (3,2), (3,1)],
          [(3,0), (2,2), (2,1), (2,0), (1,1), (1,0), (0,0)] ]
DX = 56
DY = 98

def getBoneBoard( html, board ):
    properties = html.split(';')

    bone = properties[6]
    bone = bone.split(':')[1]
    bone = bone.split('px')

    x = bone[0].strip()
    y = bone[1].strip()

    i = abs( int( x )//DX )
    j = abs( int( y )//DY )

    positions = properties[9]
    position = re.findall(r'-?\d+[.\d+]*px', positions )
    rotation = re.findall(r'-?\d+deg', positions )
    x = float( position[0][:-2] )
    deg = int( rotation[0][:-3] ) % 360
    
    # print( f'{positions}|{rotation}|{deg}' )
    # print( f'{positions}|{position}|{x}' )

    BONE =  BONES[j][i]

    if deg == 90 : BONE = BONE[::-1]

    if len(board) == 0 : board.append( BONE )
    else:
        if x > 0 : 
            if board[-1][1] != BONE[0] : BONE = BONE[::-1]
            board.append( BONE )
        else : 
            if board[0][0] != BONE[1] : BONE = BONE[::-1]
            board.insert( 0, BONE )
    
    return board

def getBoneHand( html ):
    properties = html.split(';')
    bone = next( ( p for p in properties if p.startswith(' background-position:') ), None )
    bone = re.findall(r'-?\d+[.\d+]*px', bone )

    x = abs( float( bone[-2][:-2] ) )
    y = abs( float( bone[-1][:-2] ) )

    i = int( x//DX )
    j = int( y//DY )

    return BONES[j][i]

def getBoneHand2( html ):
    properties = html.split(';')
    bone = next( ( p for p in properties if p.startswith(' background-position:') ), None )
    bone = re.findall(r'-?\d+[.\d+]*px', bone )

    x = abs( float( bone[0][:-2] ) )
    y = abs( float( bone[1][:-2] ) )

    i = int( x//DX )
    j = int( y//DY )

    return BONES[j][i]

def getUnknownBones( htmlList ) :
    u = [0,0,0]
    for html in htmlList :
        properties = html.split(';')
        positions = properties[10]
        position = re.findall(r'-?\d+[.\d+]*px', positions )
        rotation = re.findall(r'-?\d+deg', positions )

        x = float( position[0][:-2] )
        deg = int( rotation[0][:-3] ) % 360

        if deg == 0 : u[1] += 1
        else:
            if x < 100 : u[0] += 1
            else : u[2] += 1
    
    return u

def isMyTurn( htmlList ) :
    myTurn = False
    for html in htmlList :
        properties = html.split(';')
        images = properties[6]
        images = images.split(':')[1]
        images = images.split(',')

        if len( images ) == 1:
            myTurn = True
            break

    return myTurn

def areFinalBones( htmlList ) :
    for html in htmlList :
        properties = html.split(';')
        positions = properties[10]

        if 'scale' in positions : return True
    return False

def getFinalBones( htmlList , W, H ) :
    P = [ [3,2], [0,1] ]
    u = [ [],[],[],[] ]
    for html in htmlList :
        properties = html.split(';')
        positions = next( ( p for p in properties if p.startswith(' transform:') ), None )
        position = re.findall(r'-?\d+[.\d+]*px', positions )

        x = abs( float( position[0][:-2] ) )
        y = abs( float( position[1][:-2] ) )

        i1 = x/W < y/H
        i2 = x/W + y/H <= 1
        idx = P[i1][i2]

        bone = next( ( p for p in properties if p.startswith(' background-position:') ), None )
        # print(bone)
        bone = re.findall(r'-?\d+[.\d+]*px', bone )
        # print(bone)
        x = bone[-2][:-2]
        y = bone[-1][:-2]

        i = int( abs( float( x )/40 ) // 1 )
        j = int( abs( float( y )/70 ) // 1 )

        bone = BONES[j][i]
        u[idx].append( bone )    
    
    return u   

def isAnAvaliableBone( html ) :
    properties = html.split(';')
    images = next( ( p for p in properties if p.startswith(' background-image:') ), None )
    
    if len( images.split( ',' ) ) == 1 : return True
    else : return False

def play() :
    # Mute Option
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument("--mute-audio")

    # Minimize Window
    chrome_options.add_argument('--headless')

    # Open Google Chrome Browser
    driver = webdriver.Chrome("./chromedriver.exe",chrome_options=chrome_options)

    # Open Dominoes page
    driver.get("https://dominoes.playdrift.com/")

    # Select SinglePlayer
    driver.find_element_by_link_text('Singleplayer').click()

    # Select Block Type
    select = Select( driver.find_element_by_name('chain_dominoes') )
    select.select_by_value('block')

    # Select 4 Players
    select = Select( driver.find_element_by_name('size') )
    select.select_by_value('4')

    # Select 4 Players
    select = Select( driver.find_element_by_name('opponent') )
    select.select_by_value('bothard')

    ## Additional Options
    driver.find_element_by_xpath(".//*[contains(text(),'Show additional')]").click()

    # Select Limit Points
    select = Select( driver.find_element_by_name('limit') )
    select.select_by_value('150')

    # Select Initial
    select = Select( driver.find_element_by_name('bones') )
    select.select_by_value('7')

    ## Start Game
    driver.find_element_by_link_text('Play').click()

    ## Read Game
    # Board
    BOARDS = []
    BONES_NUMBER = []
    BOARD = []
    board = driver.find_element_by_class_name('board')

    # My Hand
    myHand = []
    hand = driver.find_element_by_class_name('hand.local')
    myHandFound = False

    # Other players hands
    handsList = []
    hands = driver.find_element_by_class_name('hands')

    # Game Played
    gamePlayed = False

    # My Turn
    myTurn = False
    initTime = datetime.datetime.now()

    # Ended
    end = False

    while True:
        try:
            bonesTemp = board.find_elements_by_class_name('bone')

            # If another bone is in the board...
            if len(bonesTemp) > len(BOARD) :
                gamePlayed = True

                myTurnMessage = True

                #... add it to the board
                bonesTemp = board.find_elements_by_class_name('bone')
                BOARD = getBoneBoard( dpc( bonesTemp[-1].get_attribute('style') ), BOARD )

                #... and count the unknown bones
                unknownBones = hands.find_elements_by_class_name('bone')
                u = getUnknownBones( [ dpc(b.get_attribute('style') ) for b in unknownBones  ] )
                u.insert( 0, len(hand.find_elements_by_class_name('bone')) )

                if DEBUG : print( BOARD )
                # print( u )
                # print()

                BOARDS.append( dpc( BOARD ) )
                BONES_NUMBER.append( u )

            # Are the final bones?
            myHandTemp = hand.find_elements_by_class_name('bone')
            myHandTemp = [ b.get_attribute('style') for b in myHandTemp ]

            bonesTemp = hands.find_elements_by_class_name('bone')
            bonesTemp = [ b.get_attribute('style') for b in bonesTemp ]
            finalBones = areFinalBones( bonesTemp )

            time.sleep(0.0075)
            bonesTemp = hands.find_elements_by_class_name('bone')
            bonesTemp = [ b.get_attribute('style') for b in bonesTemp ]

            if finalBones :
                # print('Últimas Fichas')

                # Window Size
                SIZE = driver.get_window_size()
                W,H = SIZE['width'], SIZE['height']

                final = getFinalBones( bonesTemp, W, H )

                myHandFinal = [ getBoneHand2( b ) for b in myHandTemp ]
                final[0] = myHandFinal

                # for b in final : print( b )

                driver.close()
                break

            # Is my turn?
            myHandTemp = hand.find_elements_by_class_name('bone')
            myTurn = isMyTurn( [ dpc( b.get_attribute('style') ) for b in myHandTemp ] )
            dT = datetime.datetime.now() - initTime

            if myTurn and myHandFound and dT.seconds > 5 and not end:
                actions = ActionChains(driver)
                initTime = datetime.datetime.now()

                # Play.... randomly
                bone = next( ( bone for bone in myHandTemp if isAnAvaliableBone( bone.get_attribute('style') ) ), None )
                actions.move_to_element( bone )
                actions.click_and_hold(on_element=None)
                actions.perform()

                highlight = board.find_elements_by_class_name('dominoesui-highlight')
                if len(highlight) : actions.move_to_element( highlight[0] )
                actions.release(on_element=None)
                actions.perform()

            # Find out my hand
            if not myHandFound:
                myHandTemp = hand.find_elements_by_class_name('bone')
                if len( myHandTemp ) == 7 :
                    myHand = [ getBoneHand( dpc( b.get_attribute('style') ) ) for b in myHandTemp ]
                    myHandFound = True
                    # print( f"My Hand: \n {myHand} \n" )

        except KeyboardInterrupt :
            driver.close()
            break

        except:
            driver.close()
            pass
            raise

    if DEBUG:
        print(myHand)
        print()
        for b,n in zip(BOARDS,BONES_NUMBER) : print(f'{b} | {n}')
        print()
        print( final )

    ## SAVE
    DATA = [ myHand, BOARDS, BONES_NUMBER, final ]
    with open( 'games/RAW_' + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M') + '.h5' ,"wb") as f:
        pickle.dump( DATA, f )

