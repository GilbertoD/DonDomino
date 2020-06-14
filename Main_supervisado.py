
import numpy as np
import pickle
import tensorflow as tf
import tensorflow.keras.layers as l
import tensorflow_addons as ta
from sklearn.model_selection import train_test_split
print(tf.__version__)
fileName = 'DATA.h5'
with open(fileName, 'rb') as r:
    DATA = pickle.load(r)

STATES, ACTIONS = DATA['s'], DATA['a']
print(f'# States {len(STATES)} # Actions {len(ACTIONS)}')

def build_model(input_shape):

    layers = [l.Conv1D(32,10,activation=ta.activations.gelu,strides=3,padding="same",input_shape= input_shape),
              ta.layers.Maxout(32),
            l.Conv1D(64,10,activation=ta.activations.gelu,strides=5),
            l.LocallyConnected1D(128,5),
            l.Flatten(),
            l.Dense(7*7,activation="softmax")
            ]

    model = tf.keras.Sequential(layers)

    model.compile(optimizer=tf.keras.optimizers.Adagrad(lr=0.0005),loss=tf.keras.losses.CategoricalCrossentropy(),metrics=tf.keras.metrics.CategoricalCrossentropy())
    model.summary()
    return model




def arreglar_datos(x_pre,y_pre):
    x_pos=[]
    y_pos=[]
    for i in range(len(x_pre)):

        temp_x =[]
        for elem in x_pre[i]:
            temp_x.extend(elem)
        x_pos.append(np.array(temp_x).reshape(-1,1))
        y_pos.append(np.array(y_pre[i]).ravel())
    return np.array(x_pos,dtype=np.int32),np.array(y_pos,dtype=np.int32)
def grad(model, inputs, targets):
  with tf.GradientTape() as tape:
    loss_value1,y1 = loss(model, inputs, targets, training=True)
  return loss_value1, tape.gradient([loss_value1],model.trainable_variables),y1

def crear_batch(x,y,batch_size):
    index = np.random.choice(range(len(x)),batch_size)
    return x[index,:],y[index,:]
def loss(model, x, y, training):
    loss_object1 = tf.keras.losses.CategoricalCrossentropy()
    y_pred = model(x,training)
    return loss_object1(y_true=y,y_pred=y_pred)

def train_model(model,X,Y,log_name,model_name,batch_size=32,step_per_epoch=10,epochs=10):
    optimizer = tf.keras.optimizers.Adadelta(learning_rate=0.001)
    # train_dataset =tf.data.Dataset.from_tensor_slices((X,Y)).batch(batch_size=batch_size).repeat().shuffle(1000)
    print("Voy a empezar el entrenamiento")
    for epoch in range(epochs):

        epoch_loss_avg1 = tf.keras.metrics.Mean()

        epoch_accuracy_start = tf.keras.metrics.CategoricalAccuracy()

        # Training loop - using batches of 32

        for i in range(step_per_epoch):
            x,y = crear_batch(X,Y,batch_size)


            loss_value1, grads,y1 = grad(model, x, y)


            optimizer.apply_gradients(zip(grads, model.trainable_variables))


            # Track progress
            epoch_loss_avg1(loss_value1)  # Add current batch loss
            # Compare predicted label to actual label

            epoch_accuracy_start(y[:,0],y1)

            # print("Log_end: " + str(y2.numpy()))
        if epoch%10 == 0:
            model.save(model_name)



        # End epoch
        # train_loss_results.append(epoch_loss_avg.result())
        # train_accuracy_results_start.append(epoch_accuracy_start.result())
        # train_accuracy_results_end.append(epoch_accuracy_end.result())

        # if epoch % 50 == 0:
        f=open(log_name,"a")
        print("Epoch {:03d}: Loss start : {:.3f},Loss end :  {:0.3f}, Accuracy_for_start: {:.3%} ,  Accuracy_for_end: {:.3%}".format(epoch,epoch_loss_avg1.result(),epoch_loss_avg2.result(),epoch_accuracy_start.result(),epoch_accuracy_end.result()))

        f.write("Epoch {:03d}: Loss start : {:.3f},Loss end :  {:0.3f}, Accuracy_for_start: {:.3%} ,  Accuracy_for_end: {:.3%}".format(epoch,epoch_loss_avg1.result(),epoch_loss_avg2.result(),epoch_accuracy_start.result(),epoch_accuracy_end.result()))
        # f.write("Log_end: " + str(y2.numpy()))
        f.close()
x,y = arreglar_datos(STATES,ACTIONS)

X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.2)
model = build_model((x[0].shape[0],x[0].shape[1],))

model_callback = tf.keras.callbacks.ModelCheckpoint("models/DOMINATOR_e{epoch}-val_loss_{val_loss:.4f}.hdf5",save_best_only=True)
history = model.fit(X_train,y_train,batch_size=1,epochs=100,verbose=1,callbacks=[model_callback])
loss_ = history["loss"]
val_loss = history["val_loss"]
model.evaluate(X_test,y_test)

