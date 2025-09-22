import numpy as np
import math
import matplotlib.pyplot as plt

def main():
    # --- Part A --- 

    # Q2:
    _Control = np.array([[0.5, 0, 1.0],
                         [0.0, -1/(2*math.pi), 1.0],
                         [0.5, 0, 1.0],
                         [0.0, 1/(2*math.pi), 1.0],
                         [0.5, 0, 1.0]]) # dt, v, omega
    _time = np.cumsum(_Control[:, 2]) - _Control[0, 2] # compute time stamps at the start of each control
    _Control = np.column_stack((_time, _Control)) # add time stamps to control
    _Control = [Control(row[0], row[1], row[2], row[3]) for row in _Control]
    
    state_list = motion_model(State(), _Control) # start at (t, x, y, theta) = (0, 0, 0, 0)

    # fig, ax = plt.subplots(figsize=(10,5))
    # ax.set_title("Q2: Robot motion model based on the 6 given control inputs")
    # plot_state(fig, ax, state_list, "Motion model")

    # Q3:
    i = 0 # dataset index
    sample_time = 0.02 # 50 Hz
    ds_Control, ds_GroundTruth, ds_Landmark_GroundTruth, ds_Measurement = import_data(i)
    ds_Control, ds_GroundTruth, ds_Measurement = resample_data(sample_time, ds_Control, ds_GroundTruth, ds_Measurement)
    
    motion_model_State = motion_model(ds_GroundTruth[0], ds_Control) # starting at first ground truth state

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
    fig.suptitle(f"Q3: Robot trajectories based on ds{i}_Control.dat")
    plot_state(fig, ax1, motion_model_State, "Motion model")
    plot_state(fig, ax1, ds_GroundTruth, "Ground truth")
    plot_landmarks(fig, ax1, ds_Landmark_GroundTruth)

    ax2.set_title(f"Zoomed in view")
    plot_state(fig, ax2, motion_model_State, "Motion model")
    plot_state(fig, ax2, ds_GroundTruth, "Ground truth")
    plot_landmarks(fig, ax2, ds_Landmark_GroundTruth)
    zoom = 1
    ax2.set_xlim(ds_GroundTruth[0].x-zoom, ds_GroundTruth[0].x+zoom)
    ax2.set_ylim(ds_GroundTruth[0].y-zoom, ds_GroundTruth[0].y+zoom)

    # Q6:
    test_State = [State(0, 2.0, 3.0, 0.0), State(0, 0.0, 3.0, 0.0), State(0, 1.0, -2.0, 0.0)]
    test_LM_id = [6, 13, 17]

    for j, state in enumerate(test_State):
        test_Landmark = [landmark for landmark in ds_Landmark_GroundTruth if landmark.id == test_LM_id[j]] # extract single landmark from list
        measurement_model_t(state, test_Landmark)
        print(f"\nRobot position (x, y, theta) = ({state.x:.2f} m, {state.y:.2f} m, {state.theta:.2f} rad)")
        for landmark in state.LM_measured:
            if landmark.id == test_LM_id[j]:
                print(f"Landmark {landmark.id} predicted at: (range, heading) = ({landmark.d:.2f} m, {landmark.alpha:.2f} rad)")
                # error = math.sqrt((landmark.x - test_Landmark[0].x)**2 + (landmark.y - test_Landmark[0].y)**2)
                # print
                break

    # fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20,10))
    # fig.suptitle("Q6: Measurement model predictions for 3 different robot positions")
    # plot_measurement_predictions(fig, (ax1, ax2, ax3), test_State, ds_Landmark_GroundTruth)

    plt.show()

### DATA IMPORT ###

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
    ds_Landmark_GroundTruth = [Landmark(row[0], row[1], row[2], row[3], row[4]) for row in ds_Landmark_GroundTruth_raw] # convert into list of Landmark objects
    
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
    
def resample_data(dt, ds_Control, ds_GroundTruth, ds_Measurement): ################ needs work
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
            robot_id_possible.add(int(i))

    if len(robot_id_possible) != 1:
        print(f"More than one possible robot: {robot_id_possible}")
        return None
    else:
        robot_id = int(robot_id_possible.pop())
        print(f"Robot id is {robot_id}")
        return robot_id

### MOTION MODEL ###

def motion_model(state_0, ds_Control): # uses initial state and a list of control objects, returns a list of state objects
    motion_model_State = []
    motion_model_State.append(state_0)
    for control in ds_Control:
        next_state = motion_model_t(motion_model_State[-1], control, process_noise=None)
        motion_model_State.append(next_state)
    return motion_model_State

def motion_model_t(state, control, process_noise=None): # uses current state and a control object, returns the next state
    x_stddev = 0.0
    y_stddev = 0.0
    theta_stddev = 0.0

    process_noise_x = np.random.normal(0, x_stddev**2)
    process_noise_y = np.random.normal(0, y_stddev**2)
    process_noise_theta = np.random.normal(0, theta_stddev**2)

    if control.omega == 0:
        new_x = state.x + control.v * math.cos(state.theta) * control.dt + process_noise_x
        new_y = state.y + control.v * math.sin(state.theta) * control.dt + process_noise_y
        new_theta = state.theta + process_noise_theta
    else:
        new_x = state.x + (control.v / control.omega) * (math.sin(state.theta + control.omega * control.dt) - math.sin(state.theta)) + process_noise_x
        new_y = state.y + (control.v / control.omega) * (math.cos(state.theta) - math.cos(state.theta + control.omega * control.dt)) + process_noise_y
        new_theta = state.theta + control.omega * control.dt + process_noise_theta

    return State(control.dt, new_x, new_y, new_theta)

### MEASUREMENT MODEL ###

def measurement_model(ds_State, ds_Landmark_GroundTruth): # uses landmark ground truth and measurements as input, returns estimated state as output
    
    for state in ds_State:
        measurement_model_t(state, ds_Landmark_GroundTruth, measurement_noise=None)

def measurement_model_t(state, ds_Landmark_GroundTruth, measurement_noise=None): # uses current state and landmark ground truth as input, adds noise to the LM, returns state with estimated landmarks as output
    for landmark in ds_Landmark_GroundTruth:
        noise_x = np.random.normal(0, landmark.x_stddev)
        noise_y = np.random.normal(0, landmark.y_stddev)
        landmark_noise = Landmark(landmark.id, landmark.x + noise_x, landmark.y + noise_y) # add noise to landmark position
        d_est = math.sqrt((landmark_noise.x - state.x)**2 + (landmark_noise.y - state.y)**2)
        alpha_est = math.atan2(landmark_noise.y - state.y, landmark_noise.x - state.x) - state.theta
        state.add_landmark(Measurement(state.t, landmark.id, d_est, alpha_est))



### PLOTTING ###

def plot_state(fig, ax, ds_State, label):
    x = [state.x for state in ds_State]
    y = [state.y for state in ds_State]
    theta = [state.theta for state in ds_State]

    ax.scatter(x[0], y[0], marker='x', label=f'Start {label}')
    ax.scatter(x[-1], y[-1], marker='*', label=f'End {label}')

    ax.plot(x, y, label=label)
    # ax.quiver(x, y, np.cos(theta), np.sin(theta), color='r', scale=20)
    ax.quiver(x[0], y[0], math.cos(theta[0]), math.sin(theta[0]), width=0.005, scale=20, color='r', label=f'Start orientation {label}')
    ax.quiver(x[-1], y[-1], math.cos(theta[-1]), math.sin(theta[-1]), width=0.005, scale=20, color='b', label=f'End orientation {label}')
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
        ax.text(landmark.x, landmark.y, f'LM{landmark.id}')


def plot_measurement_predictions(fig, axes, ds_State, ds_Landmark_GroundTruth):
    for i, state in enumerate(ds_State):
        ax = axes[i]
        ax.set_title(f"(x, y, theta) = ({state.x:.2f} m, {state.y:.2f} m, {state.theta:.2f} rad)")
        for landmark in state.LM_est:
            ax.scatter(landmark.x, landmark.y, marker='x', color='red')
            ax.text(landmark.x, landmark.y, f'LM prediction {landmark.id}')
            for landmark_gt in ds_Landmark_GroundTruth:
                if landmark.id == landmark_gt.id:
                    ax.scatter(landmark_gt.x, landmark_gt.y, marker='o', color='black')
                    ax.text(landmark_gt.x, landmark_gt.y, f'LM ground truth {landmark_gt.id}')
        ax.scatter(state.x, state.y, marker='*', color='green', label='Robot position')
        ax.quiver(state.x, state.y, math.cos(state.theta), math.sin(state.theta), width=0.005, scale=20, color='green', label='Robot orientation')
        ax.set_xlabel('X position (m)')
        ax.set_ylabel('Y position (m)')
        ax.legend()
        ax.axis('equal')
        ax.grid(True)



### DATA STRUCTURES ###

class State:
    def __init__(self, t=0.0, x=0.0, y=0.0, theta=0.0):
        self.t = t
        self.x = x
        self.y = y
        self.theta = theta
        self.LM_measured = [] # list of measured landmarks as measurement objects
        self.LM_est = [] # list of estimated landmarks as landmark objects (for plotting)

    def add_landmark(self, landmark):
        self.LM_measured.append(landmark)
        self.LM_est.append(Landmark(landmark.id, self.x + landmark.d * math.cos(landmark.alpha + self.theta), self.y + landmark.d * math.sin(landmark.alpha + self.theta)))

class Control:
    def __init__(self, t, v, omega, dt):
        self.t = t
        self.v = v
        self.omega = omega
        self.dt = dt

class Landmark:
    def __init__(self, id, x, y, x_stddev=0.0, y_stddev=0.0):
        self.id = int(id)
        self.x = x
        self.y = y
        self.x_stddev = x_stddev
        self.y_stddev = y_stddev

class Measurement:
    def __init__(self, t, id, d, alpha):
        self.t = t
        self.id = int(id)
        self.d = d
        self.alpha = alpha


if __name__ == "__main__":
    main()