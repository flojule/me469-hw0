import numpy as np
import math
import matplotlib.pyplot as plt

def main():
    # --- Part A --- 

    # Q2:
    _Control = np.array([[0.5, 0, 1.0],
                         [0.0, -1/(2*math.pi), 1.0],
                         [0.5, 0, 1.0],
                         [0.0, -1/(2*math.pi), 1.0],
                         [0.5, 0, 1.0]]) # dt, v, omega
    _time = np.cumsum(_Control[:, 2]) - _Control[0, 2] # compute time stamps at the start of each control
    _Control = np.column_stack((_time, _Control)) # add time stamps to control
    _Control = [Control(row[0], row[1], row[2], row[3]) for row in _Control]
    
    state_list = compute_states(State(0, 0, 0, 0), _Control) # start at (t, x, y, theta) = (0, 0, 0, 0)

    # fig, ax = plt.subplots(figsize=(10,10))
    # ax.set_title("Q2: Robot motion model based on the 6 given control inputs")
    # plot_state(fig, ax, state_list, "Motion model")

    # Q3:
    i = 0 # dataset index
    ds_Control, ds_GroundTruth, ds_Landmark_GroundTruth, ds_Measurement = import_data(i)
 
    state_0 = ds_GroundTruth[0] # x, y, theta starting point from ground truth (1.29812900, 1.88315210, 2.82870000)
    
    ds_State = compute_states(ds_GroundTruth[0], ds_Control) # 

    fig, ax = plt.subplots(figsize=(10,10))
    ax.set_title(f"Q3: Robot trajectories based on ds{i}_Control.dat")
    plot_state(fig, ax, ds_State, "Motion model")
    plot_state(fig, ax, ds_GroundTruth, "Ground truth")
    plot_landmarks(fig, ax, ds_Landmark_GroundTruth)

    # Q6:


    plt.show()

### DATA IMPORT FUNCTIONS ###

def import_data(i):
    ds_Control_raw = import_dat(f'ds{i}/ds{i}_Control.dat')
    ds_Control = []
    for j, control in enumerate(ds_Control_raw): # convert into list of Control objects
        dt =  ds_Control_raw[j+1][0] - control[0] if j < len(ds_Control_raw) - 1 else control[0] - ds_Control_raw[j-1][0] # compute dt, exception for last element
        control = Control(control[0], control[1], control[2], dt)
        ds_Control.append(control)
    
    ds_GroundTruth_raw = import_dat(f'ds{i}/ds{i}_GroundTruth.dat')
    ds_GroundTruth = [State(row[0], row[1], row[2], row[3]) for row in ds_GroundTruth_raw] # convert into list of State objects
    
    ds_Landmark_GroundTruth_raw = import_dat(f'ds{i}/ds{i}_Landmark_GroundTruth.dat')  
    ds_Landmark_GroundTruth = [Landmark(row[0], row[1], row[2]) for row in ds_Landmark_GroundTruth_raw] # convert into list of Landmark objects
    
    ds_Measurement_raw = import_dat(f'ds{i}/ds{i}_Measurement.dat') 
    ds_Measurement = [Measurement(row[0], row[1], row[2], row[3]) for row in ds_Measurement_raw] # convert into list of Measurement objects
    ds_Barcodes = import_dat(f'ds{i}/ds{i}_Barcodes.dat')  
    for measurement in ds_Measurement:
        for j in range(len(ds_Barcodes)):
            if ds_Barcodes[j][1] == measurement.id:
                measurement.id = int(ds_Barcodes[j][0]) # replace barcode with id
                break

    robot_id = get_robot_id(ds_Measurement, ds_Barcodes)

    return ds_Control, ds_GroundTruth, ds_Landmark_GroundTruth, ds_Measurement

def import_dat(filename):
    with open(filename, 'r') as file:
        return np.loadtxt(file)
    
def resample_data(data, dt):
    pass

def get_robot_id(ds_Measurement, ds_Barcodes): # find robot id
    ids = set()
    for measurement in ds_Measurement:
        ids.add(measurement.id)

    robot_id_pos = set()
    for i in ds_Barcodes[:, 0]:
        if i not in ids:
            robot_id_pos.add(i)

    if len(robot_id_pos) != 1:
        print(f"More than one possible robot: {robot_id_pos}")
        return None
    else:
        robot_id = int(robot_id_pos.pop())
        print(f"Robot id is {robot_id}")
        return robot_id

### MOTION MODEL FUNCTIONS ###

def motion_model(state, control): # takes in a state and control object, returns new state object
    if control.omega == 0:
        new_x = state.x + control.v * math.cos(state.theta) * control.dt
        new_y = state.y + control.v * math.sin(state.theta) * control.dt
        new_theta = state.theta
    else:
        new_x = state.x + (control.v / control.omega) * (math.sin(state.theta + control.omega * control.dt) - math.sin(state.theta))
        new_y = state.y + (control.v / control.omega) * (math.cos(state.theta) - math.cos(state.theta + control.omega * control.dt))
        new_theta = state.theta + control.omega * control.dt
    return State(control.dt, new_x, new_y, new_theta)

def compute_states(state_0, ds_Control): # takes the initial state and a list of control objects, returns a list of state objects
    ds_State = []
    ds_State.append(state_0)
    for _control in ds_Control:
        new_state = motion_model(ds_State[-1], _control)
        ds_State.append(new_state)
    return ds_State

### MEASUREMENT MODEL FUNCTIONS ###

def measurement_model(state, landmark):
    pass

### PLOTTING FUNCTIONS ###
    
def plot_state(fig, ax, data, label):
    x = [state.x for state in data]
    y = [state.y for state in data]
    theta = [state.theta for state in data]

    ax.scatter(x[0], y[0], marker='x', label=f'Start {label}')
    ax.scatter(x[-1], y[-1], marker='*', label=f'End {label}')

    ax.plot(x, y, label=label)
    # ax.quiver(x, y, np.cos(theta), np.sin(theta), color='r', scale=20)
    ax.quiver(x[0], y[0], math.cos(theta[0]), math.sin(theta[0]), width=0.005, scale=10, color='r', label=f'Start orientation {label}')
    ax.quiver(x[-1], y[-1], math.cos(theta[-1]), math.sin(theta[-1]), width=0.005, scale=10, color='b', label=f'End orientation {label}')
    ax.set_xlabel('X position (m)')
    ax.set_ylabel('Y position (m)')
    ax.legend()
    ax.axis('equal')
    ax.grid(True)

def plot_landmarks(fig, ax, ds_Landmarks):
    x = [landmark.x for landmark in ds_Landmarks]
    y = [landmark.y for landmark in ds_Landmarks]
    ax.scatter(x, y, marker='o', color='black', label='Landmarks')
    for landmark in ds_Landmarks:
        ax.text(landmark.x, landmark.y, f'ID: {landmark.id:.0f}')

### DATA STRUCTURES ###

class State:
    def __init__(self, t, x, y, theta):
        self.t = t
        self.x = x
        self.y = y
        self.theta = theta

class Control:
    def __init__(self, t, v, omega, dt):
        self.t = t
        self.v = v
        self.omega = omega
        self.dt = dt

class Landmark:
    def __init__(self, id, x, y):
        self.id = id
        self.x = x
        self.y = y

class Measurement:
    def __init__(self, t, id, range, bearing):
        self.t = t
        self.id = id
        self.range = range
        self.bearing = bearing


if __name__ == "__main__":
    main()