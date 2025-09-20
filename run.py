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
    
    state_list = motion_model(State(0, 0, 0, 0), _Control) # start at (t, x, y, theta) = (0, 0, 0, 0)

    # fig, ax = plt.subplots(figsize=(10,10))
    # ax.set_title("Q2: Robot motion model based on the 6 given control inputs")
    # plot_state(fig, ax, state_list, "Motion model")

    # Q3:
    i = 0 # dataset index
    sample_time = 0.02 # 50 Hz
    ds_Control, ds_GroundTruth, ds_Landmark_GroundTruth, ds_Measurement = import_data(i)
    ds_Control, ds_GroundTruth, ds_Measurement = resample_data(sample_time, ds_Control, ds_GroundTruth, ds_Measurement)
    
    ds_State = motion_model(ds_GroundTruth[0], ds_Control) # starting at first ground truth state

    fig, ax = plt.subplots(figsize=(10,10))
    ax.set_title(f"Q3: Robot trajectories based on ds{i}_Control.dat")
    plot_state(fig, ax, ds_State, "Motion model")
    plot_state(fig, ax, ds_GroundTruth, "Ground truth")
    plot_landmarks(fig, ax, ds_Landmark_GroundTruth)

    # Q6:

    test_State = [State(0, 2.0, 3.0, 0.0), State(0, 0.0, 3.0, 0.0), State(0, 1.0, -2.0, 0.0)]
    test_LM_id = [6, 13, 17]
    test_Landmark = [[landmark for landmark in ds_Landmark_GroundTruth if landmark.id == test_LM_id[0]],
                     [landmark for landmark in ds_Landmark_GroundTruth if landmark.id == test_LM_id[1]],
                     [landmark for landmark in ds_Landmark_GroundTruth if landmark.id == test_LM_id[2]]]



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
    
def resample_data(dt, ds_Control, ds_GroundTruth, ds_Measurement):
    # measurements rounded to nearest timestep
    # Control and GroundTruth linearly interpolated to fixed timestep

    max_time = min(ds_Control[-1].t, ds_GroundTruth[-1].t) # find the maximum time that is available in all datasets
    min_time = min(ds_Control[0].t, ds_GroundTruth[0].t) # find the minimum time that is available in all datasets
    timesteps = int((max_time - min_time) / dt) + 1

    return ds_Control, ds_GroundTruth, ds_Measurement
    

def get_robot_id(ds_Measurement, ds_Barcodes): # find robot id
    ids = set()
    ids.update(measurement.id for measurement in ds_Measurement)

    robot_id_possible = set()
    for i in ds_Barcodes[:, 0]:
        if i not in ids:
            robot_id_possible.add(i)

    if len(robot_id_possible) != 1:
        print(f"More than one possible robot: {robot_id_possible}")
        return None
    else:
        robot_id = int(robot_id_possible.pop())
        print(f"Robot id is {robot_id}")
        return robot_id

### MOTION MODEL FUNCTIONS ###

def motion_model(state_0, ds_Control): # takes in the initial state and a list of control objects, returns a list of state objects
    ds_State = []
    ds_State.append(state_0)
    for _control in ds_Control:
        if _control.omega == 0:
            new_x = ds_State[-1].x + _control.v * math.cos(ds_State[-1].theta) * _control.dt
            new_y = ds_State[-1].y + _control.v * math.sin(ds_State[-1].theta) * _control.dt
            new_theta = ds_State[-1].theta
        else:
            new_x = ds_State[-1].x + (_control.v / _control.omega) * (math.sin(ds_State[-1].theta + _control.omega * _control.dt) - math.sin(ds_State[-1].theta))
            new_y = ds_State[-1].y + (_control.v / _control.omega) * (math.cos(ds_State[-1].theta) - math.cos(ds_State[-1].theta + _control.omega * _control.dt))
            new_theta = ds_State[-1].theta + _control.omega * _control.dt

        ds_State.append(State(_control.dt, new_x, new_y, new_theta))
    return ds_State

### MEASUREMENT MODEL FUNCTIONS ###

def measurement_model(ds_State, ds_Measurement):
    for state in ds_State:
        for measurement in ds_Measurement:
            if abs(measurement.t - state.t) < 1e-5: # only consider measurements at the same time as the state
                x = measurement.d * math.cos(state.theta + measurement.alpha) + state.x
                y = measurement.d * math.sin(state.theta + measurement.alpha) + state.y
                state.add_landmark(Landmark(measurement.id, x, y))




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
        ax.text(landmark.x, landmark.y, f'LM{landmark.id:.0f}')

### DATA STRUCTURES ###

class State:
    def __init__(self, t, x, y, theta):
        self.t = t
        self.x = x
        self.y = y
        self.theta = theta
        self.LM = [] # list of measured landmarks

    def add_landmark(self, landmark):
        self.LM.append(landmark)

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
    def __init__(self, t, id, d, alpha):
        self.t = t
        self.id = id
        self.d = d
        self.alpha = alpha


if __name__ == "__main__":
    main()