from itertools import combinations 
import random as rnd
from copy import deepcopy as dpc
import definitions as defs
import numpy as np



DEBUG = False

class Juego :

    def __init__(self, nMax:int, nJug:int,prueba:bool) :
        self.nMax = nMax
        assert self.cantFichas() % nJug == 0, "No se pueden repartir las fichas!!!"
        self.nJug = nJug

        nFichas = int(self.cantFichas()/nJug)
        types = ['no random','random','random','random']
        self.jugadores = [ Jugador( i, nFichas, nMax, types[i] ) for i in range(nJug) ]

        self.tablero = []
        self.fichas = []

        cantFichas = self.cantFichas()
        self.policy = defs.Policy( cantFichas*nJug + nMax + 1, 2*cantFichas + 1 )
        self.prueba = prueba

    def cantFichas(self) -> int:
        n = self.nMax
        return int( 0.5*( n + 1 )*( n + 2 ) )

    def repartir(self):
        numeros = range(self.nMax+1)

        fichas = list( combinations( numeros, 2) )
        for i in numeros: fichas.append( (i,i) )

        jugIdx = [ i % self.nJug for i in range (len( fichas ) ) ]
        rnd.shuffle( jugIdx )
        for ficha, jug in zip( fichas, jugIdx ): 
            f = Ficha(ficha[0],ficha[1])
            self.fichas.append( dpc( f ) )
            self.jugadores[jug].agregarFicha( f )
    
    def reset(self):
         self.tablero = []
         self.fichas = []
         for jugador in self.jugadores : jugador.fichas=[]

    def printJugadores(self):
        for j in self.jugadores: print(j)

    def printTablero(self):
        s = f'Tablero:\n\t'
        for ficha in self.tablero: s += str(ficha) + "  "
        print(s+"\n")

    def jugar(self):
        self.repartir()
        if DEBUG : self.printJugadores()            

        idx = -1
        for i,j in enumerate(self.jugadores):
            if Ficha(self.nMax,self.nMax) in j.fichas:
                idx = i
                break

        k = 0
        nPas = 0
        acabar = False

        self.encFichas = Encoder( self.fichas )
        self.encNum = Encoder( range(self.nMax+1) )

        while not acabar:
            self.tablero, ficha, acabar, pasar = self.jugadores[idx].jugar( self.tablero, self )

            if DEBUG : print(f'Turno {k:d}, el Jugador {idx:d} juega la Ficha {ficha}')

            if pasar: nPas += 1
            else: nPas = 0

            if nPas == self.nJug: acabar = True

            if DEBUG : self.printJugadores() 
            if DEBUG : self.printTablero()

            idx += 1
            idx %= self.nJug
            k += 1

        idx = (idx-1)%self.nJug

        # self.rewards = dpc( self.policy.reward_episode )

        # if len(self.rewards) > 0 : update_policy(self.policy, self.optim)

        states, actions = [], []
        for jugador in self.jugadores : 
            states.extend( jugador.states )
            actions.extend( jugador.actions )
        actions =np.array(actions)
        if not  self.prueba: 
            self.policy.update_policy_supervised( np.array(states,dtype=np.float32), np.array(actions) )
        '''if len(self.policy.loss_history)>1000 and self.policy.loss_history[-1]>self.policy.loss_history[-100]:
            self.policy.saveModel("supervisado_mejorado")
            raise Exception("La funcion de error dejó de mejorar")
    '''

        if DEBUG and nPas < self.nJug: print(f'Se acabó el Juego, ganó {idx:d}!!!')
        if DEBUG and nPas == self.nJug: print('Se cerró el Juego :(:V')

        return self.policy.loss_history
        
class Jugador :

    def __init__(self, id:int, nFichas:int, nMax:int, typeAgent):
        self.id = id
        self.fichas = []
        self.nMax = nMax
        self.jugadas_mano=0
        self.jugada_NM=0
        self.jugadas_salto=0
        self. jugada_NM_lado_malo=0
        
        self.jugadas_buenas=0
        self.jugadas_totales=0
        
        self.typeAgent = typeAgent

        self.states = []
        self.actions = []

    def __str__(self):
        s = f'Jugador {self.id:d}:\n\t'
        for ficha in self.fichas: s += str(ficha) + "  "
        return s

    def agregarFicha( self, ficha ):
        self.fichas.append( ficha )

    def jugarRandom( self, tablero ):

        ficha = None
        nJug1, nJug2 = tablero[0].n1, tablero[-1].n2
        idx1, idx2 = False, False

        for f in self.fichas:
            if nJug1 in f : ficha, idx1 = f, True
            if nJug2 in f : ficha, idx2 = f, True
            if idx1 or idx2: break

        if ficha is not None:
            self.fichas.remove( ficha )
            if idx1: 
                if ficha.n2 == nJug1: tablero = [ficha] + tablero
                else : tablero = [ficha.inv()] + tablero
            else :
                if ficha.n1 == nJug2: tablero = tablero + [ficha]
                else : tablero = tablero + [ficha.inv()]

        return tablero, ficha, len( self.fichas ) == 0, ficha is None

    def jugar( self, tablero, juego ) :

        # Al inicio siempre se juega el [6|6]
        if not tablero:
            ficha = Ficha(self.nMax, self.nMax)
            self.fichas.remove(ficha)
            tablero.append(ficha)

            return tablero, ficha, len(self.fichas) == 0, ficha is None

        # Fichas de todos los jugadores
        fichas = [ juego.jugadores[ (self.id + i)%juego.nJug ].fichas for i in range( juego.nJug ) ]
        
        # Encoders
        encoder_to_fichas = juego.encFichas
        encoder_number = juego.encNum

        # Posibles Jugadas
        nJug1, nJug2 = tablero[0].n1, tablero[-1].n2

        # Lado para poner ficha
        idx1, idx2 = False, False

        # Ficha a poner
        ficha = None  

        ## Random Player
        if True : 

            for f in self.fichas:
                if nJug1 in f : ficha, idx1 = f, True
                if nJug2 in f : ficha, idx2 = f, True
                if idx1 or idx2: break

            if ficha is not None:
                self.fichas.remove( ficha )
                if idx1: 
                    if ficha.n2 == nJug1: tablero = [ficha] + tablero
                    else : tablero = [ficha.inv()] + tablero
                else :
                    if ficha.n1 == nJug2: tablero = tablero + [ficha]
                    else : tablero = tablero + [ficha.inv()] 

            action = -1                     
            if ficha is None : action = [ 2*juego.cantFichas() ]
            else : action = [ np.argmax(encoder_to_fichas.encode( [ficha] ) ) ]
            self.actions.append( action )

            state = []
            for f in fichas: state.extend( encoder_to_fichas.encode( f ) )
            state.extend( encoder_number.encode( [nJug1, nJug2] ) )
            state = np.array( state )
            # state = torch.tensor( state, dtype=torch.float )
            # state = state.reshape( [-1,1] )
            self.states.append( state )
            #print(juego.policy.state_space)
            state = np.array(state,dtype=np.float32).reshape((1,len(state)))
            
            action_policy = np.argmax(juego.policy.model(state))
            if action_policy == 2*juego.cantFichas() and ficha is None:
                self.jugadas_buenas+=1
                self.jugadas_salto+=1

            f=open("jugadas.txt","a")
            f.write(str(action_policy)+"\n")
            f.close
            lado = action_policy //juego.cantFichas()
            fic = np.zeros( juego.cantFichas() )
            fic[ action_policy%juego.cantFichas() ] = 1
            ficha_policy = encoder_to_fichas.decode( fic )[0]
           
          

            tengo = False
            for ff in self.fichas:
                tengo = ff == ficha_policy
                if tengo : break
            if not tengo and (nJug1 in ficha_policy or nJug2 in ficha_policy) :
                self.jugada_NM+=1
            if tengo:
                
                if lado==0 and nJug1 in  ficha_policy:self.jugadas_buenas+=1
                if lado==1 and nJug2 in  ficha_policy:self.jugadas_buenas+=1
                else:
                    self.jugadas_mano+=1

            self.jugadas_totales+=1

            return tablero, ficha, len( self.fichas ) == 0, ficha is None

        ## Código para Policy Gradient
        else :    
            # Crear Estado
            # state = encoder_to_fichas.encode( tablero )
            state = []
            for ficha in fichas: state.extend( encoder_to_fichas.encode( ficha ) )
            state.extend( encoder_number.encode( [nJug1, nJug2] ) )
            state = np.array( state )

            # Evaluar Estado
            state = juego.policy( Variable( torch.tensor( state, dtype=torch.float ) ) )

            c = Categorical(state)
            action = torch.tensor( [ c.sample().item() ], dtype=torch.int )

            lado = action // 28
            fic = np.zeros( 28 )
            fic[ action%28 ] = 1
            ficha_jugada = encoder_to_fichas.decode( fic )[0]

            tengo = False
            for ficha in self.fichas:
                tengo = ficha == ficha_jugada
                if tengo : break

            reward = 0
            if not tengo: reward -= 30
            
            print( f'Tablero: {tablero} ')
            print( f'Mis Fichas: {self.fichas}' )
            print( f'\t Tratar de jugar ficha {ficha_jugada} - Reward: {reward:d}' )

            idx1, idx2 = nJug1 in ficha, nJug2 in ficha_jugada
            
            '''if int(lado) == 0:
                if nJug1 in ficha_jugada:pass
                else: reward -= 10
            else:
                if nJug2 in ficha_jugada:pass
                else: reward -= 10'''


            # Add log probability of our chosen action to our history
            if len( juego.policy.policy_history ) != 0:
                # juego.policy.policy_history = torch.cat( [juego.policy.policy_history, torch.Tensor( [c.log_prob(action)] ) ] )
                juego.policy.reward_episode.append(reward)
            else:
                # juego.policy.policy_history = ( c.log_prob(action) )
                juego.policy.reward_episode.append( reward )

            if reward < 0:
                return tablero,ficha_jugada,True,ficha_jugada is None
                '''for f in self.fichas:
                    if nJug1 in f: ficha_jugada, idx1 = f, True
                    if nJug2 in f: ficha_jugada, idx2 = f, True
                    if idx1 or idx2: break

                action = torch.Tensor( [np.argmax(encoder_to_fichas.encode([ficha_jugada]))] )
                log=c.log_prob(action)
                numpy_history = self.policy.policy_history
                # print(self.policy.policy_history)
                self.policy.policy_history = torch.cat((self.policy.policy_history,log))
                self.policy.reward_episode.append(0)'''

            if idx1 or idx2:
                self.fichas.remove(ficha)
                if idx1:
                    if ficha.n2 == nJug1:
                        tablero = [ficha] + tablero
                    else:
                        tablero = [ficha.inv()] + tablero
                else:
                    if ficha.n1 == nJug2:
                        tablero = tablero + [ficha]
                    else:
                        tablero = tablero + [ficha.inv()]

                return tablero,ficha_jugada,len(self.fichas) == 0,ficha_jugada is None
            
            else : return tablero,ficha_jugada,True,ficha_jugada is None


class Ficha :
    def __init__(self, n1:int, n2:int):
        self.n1 = n1
        self.n2 = n2

    def __str__(self):
        return f'[{self.n1:d}|{self.n2:d}]'

    __repr__ = __str__

    def __eq__(self, value):
        if not isinstance(value, Ficha): return False
        return (self.n1 == value.n1 and self.n2 == value.n2) or (self.n1 == value.n2 and self.n2 == value.n1)

    def __contains__(self, key):
        return self.n1 == key or self.n2 == key

    def inv(self):
        self.n1, self.n2 = self.n2, self.n1
        return self

class Encoder : 
    def __init__(self, elements):
        self.elements = elements

    def encode( self, newElements ) :
        e = len(self.elements)*[0]
        for ne in newElements : e[ self.elements.index( ne ) ] = 1
        return e

    def decode( self, codeElements ) :
        e = []
        for i,ce in enumerate(codeElements) : 
            if ce == 1 : e.append( self.elements[i] )
        return e
