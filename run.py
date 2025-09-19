import numpy as np
import math
import matplotlib.pyplot as plt

def main():
    # --- Part A --- 

    # Q2:
    _Control = np.array([[1.0, 0.5, 0],
                         [1.0, 0.0, -1/(2*math.pi)],
                         [1.0, 0.5, 0],
                         [1.0, 0.0, -1/(2*math.pi)],
                         [1.0, 0.5, 0]]) # dt, v, omega
    
    state_0 = State(0, 0, 0) # x, y, theta starting point (report parameters)
    state_list = []
    state_list.append(state_0)
    state_list = compute_states(state_list, _Control)

    # fig, ax = plt.subplots(figsize=(10,10))
    # ax.set_title("Q2: Robot motion model based on the 6 given control inputs")
    # plot_state(fig, ax, state_list, "Motion model")

    # Q3:
    ds0_Control = import_dat('ds0/ds0_Control.dat')
    ds0_GroundTruth = import_dat('ds0/ds0_GroundTruth.dat')
    ds0_GroundTruth = [State(row[1], row[2], row[3]) for row in ds0_GroundTruth]

    ds1_Control = import_dat('ds1/ds1_Control.dat')
    ds1_GroundTruth = import_dat('ds1/ds1_GroundTruth.dat')
    ds1_GroundTruth = [State(row[1], row[2], row[3]) for row in ds1_GroundTruth]

    ds_Control_list = [ds0_Control, ds1_Control]
    ds_GroundTruth_list = [ds0_GroundTruth, ds1_GroundTruth]

    for i in range(len(ds_Control_list)):
        ds_Control = ds_Control_list[i]
        ds_GroundTruth = ds_GroundTruth_list[i]

        state_0 = ds_GroundTruth[0] # x, y, theta starting point from ground truth (1.29812900, 1.88315210, 2.82870000)
        ds_State = []
        ds_State.append(state_0)
        ds_State = compute_states(ds_State, ds_Control)

        fig, ax = plt.subplots(figsize=(10,10))
        ax.set_title(f"Q3: Robot trajectories based on ds{i}_Control.dat")
        plot_state(fig, ax, ds_State, "Motion model")
        plot_state(fig, ax, ds_GroundTruth, "Ground truth")

    plt.show()

def motion_model(state, control, dt):
    if control.omega == 0:
        new_x = state.x + control.v * math.cos(state.theta) * dt
        new_y = state.y + control.v * math.sin(state.theta) * dt
        new_theta = state.theta
    else:
        new_x = state.x + (control.v / control.omega) * (math.sin(state.theta + control.omega * dt) - math.sin(state.theta))
        new_y = state.y + (control.v / control.omega) * (math.cos(state.theta) - math.cos(state.theta + control.omega * dt))
        new_theta = state.theta + control.omega * dt
    return State(new_x, new_y, new_theta)
        
def compute_states(state_list, _Control):
    for i, _control in enumerate(_Control):
        dt =  _Control[i+1][0] - _control[0] if i < len(_Control) - 1 else _control[0] - _Control[i-1][0]
        control = Control(_control[1], _control[2], dt)
        new_state = motion_model(state_list[-1], control, dt)
        state_list.append(new_state)
    return state_list


def import_dat(filename):
    with open(filename, 'r') as file:
        return np.loadtxt(file)
    
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

class State:
    def __init__(self, x, y, theta):
        self.x = x
        self.y = y
        self.theta = theta

class Control:
    def __init__(self, v, omega, dt):
        self.v = v
        self.omega = omega
        self.dt = dt

if __name__ == "__main__":
    main()