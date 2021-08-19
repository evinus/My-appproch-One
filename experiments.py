from sklearn import model_selection
from tensorflow.keras.layers import Dense,Conv3D,MaxPooling3D,BatchNormalization,Flatten,Input, Add
from tensorflow.keras.models import Model , Sequential
from tensorflow import config

gpus = config.experimental.list_physical_devices('GPU') 
config.experimental.set_memory_growth(gpus[0], True)

model = Sequential()

model.add(Conv3D(input_shape =(240, 360, 3, 1),activation="relu",filters=64,kernel_size=3,padding="same"))
model.add(MaxPooling3D(pool_size=(4,4,1),strides=(2,2,1)))
model.add(BatchNormalization())

model.add(Conv3D(activation="relu",filters=128,kernel_size=3,padding="same"))
model.add(MaxPooling3D(pool_size=(2,2,1)))

model.add(Conv3D(activation="relu",filters=128,kernel_size=2,padding="same"))
model.add(MaxPooling3D(pool_size=(2,2,1)))

model.add(Flatten())
model.add(Dense(128,activation="relu"))
model.add(Dense(64,activation="relu"))
model.add(Dense(10,activation="relu"))
model.add(Dense(1,activation="sigmoid"))
model.compile()
model.summary()
from tensorflow.keras.utils import plot_model
plot_model(model,show_shapes=True)

""" input = Input((240,360,3,1))

x = Conv3D(64,3,padding="same")(input)
x = MaxPooling3D(pool_size=(3,3,3))(x)
x = Flatten()(x)
x = Dense(128)(x)

y = Dense(1)(input)
y = Flatten()(y)
y = Dense(128)(y)
y = Dense(128)(y)

x = Add()([x,y])
x = Dense(10)(x)
x = Dense(1)(x)

model = Model(inputs = input,outputs = x)
model.compile()
model.summary()
from tensorflow.keras.utils import plot_model
plot_model(model,show_shapes=True) """

""" 
model = Sequential()


model.add(Dense(1,input_shape =(240, 360, 3, 1)))
#model.add(Dense(1))
model.add(Flatten())
model.add(Dense(64))
model.add(Dense(10))
model.add(Dense(1))

model.compile()
model.summary()
from tensorflow.keras.utils import plot_model
plot_model(model,show_shapes=True) """