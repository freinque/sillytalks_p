import numpy as np

DIM_STATE = 41
DIM_RED_STATE = 9
DIM_ACTION = 18
DIM_RED_ACTION = 7
DIM_TRIVIAL_CONTROL = 3                 #trivial proportional control on x first reduced dimensions   

K_0 = 3.                                #constante de rappel, torso
K_1 = 3.                                #constante de rappel, ankle
K_2 = K_1                               #constante de rappel, other ankle
K = np.array([K_0, K_1, K_2])

C_Y_CM_TARGET = 15.                     # targets for proportional control
Y_CM_TARGET = .95
C_TORSO_ANGLE_TARGET = 7.
TORSO_ANGLE_TARGET = 0.05
ANKLE_ANGLE_TARGET = 0.00


## reduce_state in np.array form
REDUCE_STATE_W = np.zeros(shape=(DIM_STATE, DIM_RED_STATE))
REDUCE_STATE_W[22][0] = 1.
REDUCE_STATE_W[24][0] = -1.
REDUCE_STATE_W[6][1] = 1.
REDUCE_STATE_W[7][2] = 1.
REDUCE_STATE_W[8][3] = 1.
REDUCE_STATE_W[9][4] = 1.
REDUCE_STATE_W[10][5] = 1.
REDUCE_STATE_W[11][6] = 1.
REDUCE_STATE_W[19][7] = 1.
REDUCE_STATE_W[20][8] = 1.
REDUCE_STATE_B = np.zeros(shape=(DIM_RED_STATE))


## projection of red_state on first components in np.array form
TRIVIAL_RED_STATE_W = np.zeros(shape=(DIM_RED_STATE,DIM_TRIVIAL_CONTROL))
for i in range(DIM_TRIVIAL_CONTROL):
    TRIVIAL_RED_STATE_W[i][i] = 1.
TRIVIAL_RED_STATE_B = np.zeros(shape=(DIM_TRIVIAL_CONTROL))


## projection of red_state on other components in np.array form
RESIDUAL_RED_STATE_W = np.zeros(shape=(DIM_RED_STATE,DIM_RED_STATE-DIM_TRIVIAL_CONTROL))
for i in range(DIM_RED_STATE-DIM_TRIVIAL_CONTROL):
    RESIDUAL_RED_STATE_W[DIM_TRIVIAL_CONTROL+i][i] = 1.
RESIDUAL_RED_STATE_B = np.zeros(shape=(DIM_RED_STATE-DIM_TRIVIAL_CONTROL))


## proportional control on first red components in np.array form
TRIVIAL_CONTROL_W = np.zeros(shape=(DIM_TRIVIAL_CONTROL,DIM_TRIVIAL_CONTROL))
for i in range(DIM_TRIVIAL_CONTROL):
    TRIVIAL_CONTROL_W[i][i] = -K[i]
TRIVIAL_CONTROL_B = np.zeros(shape=(DIM_TRIVIAL_CONTROL))
TRIVIAL_CONTROL_B[0] = TORSO_ANGLE_TARGET*K[0]


## zero control on other red components in np.array form
RESIDUAL_CONTROL_W = np.zeros(shape=(DIM_RED_STATE-DIM_TRIVIAL_CONTROL,DIM_RED_ACTION-DIM_TRIVIAL_CONTROL))
RESIDUAL_CONTROL_B = np.zeros(shape=(DIM_RED_ACTION-DIM_TRIVIAL_CONTROL))


## inverse_reduce_action in np.array form
INVERSE_REDUCE_ACTION_W = np.zeros(shape=(DIM_RED_ACTION, DIM_ACTION))
INVERSE_REDUCE_ACTION_W[0][2] = -1.
INVERSE_REDUCE_ACTION_W[0][4] = 1.
INVERSE_REDUCE_ACTION_W[1][6] = 1.
INVERSE_REDUCE_ACTION_W[1][7] = 1.
INVERSE_REDUCE_ACTION_W[1][8] = -1.
INVERSE_REDUCE_ACTION_W[2][6+9] = 1.
INVERSE_REDUCE_ACTION_W[2][7+9] = 1.
INVERSE_REDUCE_ACTION_W[2][8+9] = -1.
INVERSE_REDUCE_ACTION_W[3][1] = -1.
INVERSE_REDUCE_ACTION_W[3][5] = 1.
INVERSE_REDUCE_ACTION_W[4][1+9] = -1.
INVERSE_REDUCE_ACTION_W[4][5+9] = 1.
INVERSE_REDUCE_ACTION_W[5][0] = 1.
INVERSE_REDUCE_ACTION_W[5][3] = -1.
INVERSE_REDUCE_ACTION_W[6][0+9] = 1.
INVERSE_REDUCE_ACTION_W[6][3+9] = -1.
INVERSE_REDUCE_ACTION_B = np.zeros(shape=(DIM_ACTION))


## reduce_action in np.array form
REDUCE_ACTION_W = INVERSE_REDUCE_ACTION_W.T
REDUCE_ACTION_B = np.zeros(shape=(DIM_RED_ACTION))


## proportional control in np.array form
PROP_CONTROL_W = np.zeros(shape=(DIM_RED_STATE, DIM_RED_ACTION))
PROP_CONTROL_W[0][0] = -K[0]
PROP_CONTROL_W[1][1] = -K[1]
PROP_CONTROL_W[2][2] = -K[2]
PROP_CONTROL_B = np.zeros(shape=(DIM_RED_ACTION))


