


import numpy as np
import random

from sklearn.linear_model import LinearRegression

import tensorflow as tf
from tensorflow.keras.layers import Dense, Activation, Flatten



#import warnings
#warnings.filterwarnings(action="ignore", category=Warning )
#warnings.simplefilter("ignore")



N = 1000

X_List = []
Y_List = []

for i in range(N) :
    A = random.randint( -10, 10 )
    B = random.randint( -10, 10 )
    Y_Val = 4*A - 3*B + 2 + random.normalvariate(0,1)
    
    X_List.append([A,B])
    Y_List.append( Y_Val )



X = np.array( X_List, dtype="f" )
Y = np.array( Y_List, dtype="f" )



regModel = LinearRegression()
regModel.fit( X, Y  )
A = round( regModel.coef_[0], 3 )
B = round( regModel.coef_[1], 3 )
INTERCEPT = round( regModel.intercept_, 3 )

print("REGRESSION")
print( "A   =", A )
print( "B   =", B )
print( "Bias=", INTERCEPT )

X_NEW = np.array( [[1,1]] )
Y_NEW = regModel.predict( X_NEW )
Y_NEW = np.round_( Y_NEW, 3 )
print( "X=",X_NEW[0] )
print( "Y=",Y_NEW[0] )

print("\n\n\n")







theShapeSize = X.shape[1]
theActivation = tf.keras.activations.linear
theLossMetric = tf.keras.losses.MeanSquaredError()
theOptimizer = tf.keras.optimizers.Adam()
theEpochs = 1000

LAYER_01 = tf.keras.layers.Dense( units=1, activation=theActivation, input_dim=theShapeSize )

##model = tf.keras.Sequential()
##model.add( LAYER_01 )
##model.compile( loss=theLossMetric,optimizer=theOptimizer)
##model.fit( X, Y, epochs=theEpochs, verbose=False )
##
##W = LAYER_01.get_weights()
##print("TENSOR FLOW")
##print( "A   =",round(W[0][0][0],3) )
##print( "B   =",round(W[0][1][0],3) )
##print( "Bias=",round(W[1][0]   ,3) )
##
##X_NEW = np.array( [[1,1]] )
##Y_NEW = model.predict( X_NEW )
##Y_NEW = np.round_( Y_NEW, 3 )
##print( "X=",X_NEW[0] )
##print( "Y=",Y_NEW[0] )
##print("\n\n\n")





print("REGRESSION\n\n")
for i in range( 10 ):
    regModel = LinearRegression()
    regModel.fit( X, Y  )
    A = round( regModel.coef_[0], 3 )
    B = round( regModel.coef_[1], 3 )
    INTERCEPT = round( regModel.intercept_, 3 )

    print("ITERATION ",i)
    print( "A   =", A )
    print( "B   =", B )
    print( "Bias=", INTERCEPT )

    X_NEW = np.array( [[1,1]] )
    Y_NEW = regModel.predict( X_NEW )
    Y_NEW = np.round_( Y_NEW, 3 )
    print( "X=",X_NEW[0] )
    print( "Y=",Y_NEW[0] )

    print("\n")



print("TENSOR FLOW\n\n")
for i in range( 10 ):
    model = tf.keras.Sequential()
    model.add( LAYER_01 )
    model.compile( loss=theLossMetric,optimizer=theOptimizer)
    model.fit( X, Y, epochs=theEpochs, verbose=False )   

    W = LAYER_01.get_weights()
    print("ITERATION ",i)
    print( "A   =",round(W[0][0][0],3) )
    print( "B   =",round(W[0][1][0],3) )
    print( "Bias=",round(W[1][0]   ,3) )

    X_NEW = np.array( [[1,1]] )
    Y_NEW = model.predict( X_NEW )
    Y_NEW = np.round_( Y_NEW, 3 )
    print( "X=",X_NEW[0] )
    print( "Y=",Y_NEW[0] )
    print("\n")





