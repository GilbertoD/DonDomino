import matplotlib.pyplot as plt
from definitions import *
from domino import *
tf.enable_eager_execution()
EPISODES = 1000

juego = Juego(6,4,False)

E = []
R = []
loss = []
# juego.policy.load_Model( 'models/supervisado_1.h5' )
open("loss.txt","w").close()
open("jugadas.txt","w").close()

for episode in range(EPISODES):
    print(f'Partida {episode+1:d}/{EPISODES:d}...')
    juego.jugar()

    juego.reset()
juego.policy.saveModel("supervisado_1")

total_buenas=0
total_en_mano=0
total_no_mano=0
total_total=0
salto=0
for jug in juego.jugadores:
    total_buenas+=jug.jugadas_buenas
    total_en_mano+=jug.jugadas_mano
    total_no_mano+=jug.jugada_NM
    total_total+=jug.jugadas_totales
    salto+=jug.jugadas_salto

print( f'Hizo {total_buenas:d} buenas jugadas de {total_total:d} jugadas totales Accuracy: {100*total_buenas/total_total:.2f}.\
    \nJugadas que no servían pero  tenía en mano {total_en_mano:d} porcentaje:{total_en_mano/total_total:0.2f}.\
    \nJugadas que eran válidas pero no las tenia {total_no_mano:d} porcentaje {total_no_mano/total_total:0.2f} \
    \n Jugadas donde salto bien saltado {salto:d} porcetaje: {salto/total_total*100:0.2f}' )



'''
plt.figure()
plt.scatter( E, R )
plt.xlabel("Episodes")
plt.ylabel("Rewards")
loss=np.loadtxt("jugadas.txt")
'''
loss=np.loadtxt("loss.txt")
plt.figure()
plt.plot( loss )
plt.xlabel("iteraciones")
plt.ylabel("Perdida")
del  loss
loss=np.loadtxt("jugadas.txt")
plt.figure()
plt.hist(loss)
del loss
plt.show()
