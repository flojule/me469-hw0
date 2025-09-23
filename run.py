import numpy as np
import math
import matplotlib.pyplot as plt

def main():
    # --- FLAGS ---
    q2 = False
    q3 = False
    q6 = False
    q7 = True

    # --- DATA IMPORT ---
    i = 0 # dataset index

    # --- Part A --- 
    ds_Control, ds_GroundTruth, ds_Landmark_GroundTruth, ds_Measurement = import_data(i)

    # Q2:
    if q2:
        _Control = np.array([[0.5, 0, 1.0],
                            [0.0, -1/(2*math.pi), 1.0],
                            [0.5, 0, 1.0],
                            [0.0, 1/(2*math.pi), 1.0],
                            [0.5, 0, 1.0]]) # v, omega, dt
        _time = np.cumsum(_Control[:, 2]) - _Control[0, 2] # compute time stamps at the start of each control
        _Control = np.column_stack((_time, _Control)) # add time stamps to control (first column)
        _Control = [Control(row[0], row[1], row[2], row[3]) for row in _Control]
        
        DR_State = dead_reckoning(State(x=(0,0,0)), _Control) # start at (x, y, theta) = (0, 0, 0)
        fig, ax = plt.subplots(figsize=(10,5))
        ax.set_title("Q2: Robot trajectories, dead reckoning, based on the 6 given control inputs")
        plot_state(fig, ax, DR_State, "Motion model")

    # Q3:
    if q3:
        DR_State = dead_reckoning(ds_GroundTruth[0], ds_Control) # starting at first ground truth state

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
        fig.suptitle(f"Q3: Robot trajectories, dead reckoning, based on ds{i}_Control.dat")
        plot_state(fig, ax1, DR_State, "Motion model")
        plot_state(fig, ax1, ds_GroundTruth, "Ground truth")
        plot_landmarks(fig, ax1, ds_Landmark_GroundTruth)

        ax2.set_title(f"Zoomed in view")
        plot_state(fig, ax2, DR_State, "Motion model")
        plot_state(fig, ax2, ds_GroundTruth, "Ground truth")
        plot_landmarks(fig, ax2, ds_Landmark_GroundTruth)
        zoom = 1
        ax2.set_xlim(ds_GroundTruth[0].x[0]-zoom, ds_GroundTruth[0].x[0]+zoom)
        ax2.set_ylim(ds_GroundTruth[0].x[1]-zoom, ds_GroundTruth[0].x[1]+zoom)

    # Q6:
    if q6:
        test_State = [State(x=[2.0, 3.0, 0.0]), State(x=[0.0, 3.0, 0.0]), State(x=[1.0, -2.0, 0.0])]
        test_LM_id = [6, 13, 17]

        for j, state in enumerate(test_State):
            test_Landmark = [landmark for landmark in ds_Landmark_GroundTruth if landmark.id == test_LM_id[j]] # extract single landmark from list
            z_est = measurement_model(state, test_Landmark)
            print(f"\nRobot position: \n(x, y, theta) = ({state.x[0]:.3f} m, {state.x[1]:.3f} m, {state.x[2]:.3f} rad)")
            for measurement in z_est:
                if measurement.id == test_LM_id[j]:
                    print(f"Landmark {measurement.id} predicted at: \n(range, bearing) = ({measurement.range:.3f} m, {measurement.bearing:.3f} rad)")
                    x, y = get_xy_measurement(state, measurement)
                    print(f"(x, y) = ({x:.3f} m, {y:.3f} m)")
                    x_gt, y_gt = test_Landmark[0].x[0], test_Landmark[0].x[1]
                    print(f"Landmark {test_Landmark[0].id} ground truth at: \n(x, y) = ({x_gt:.3f} m, {y_gt:.3f} m)")
                    break
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20,10))
        fig.suptitle("Q6: Measurement model predictions for 3 different robot positions")
        plot_measurement_predictions(fig, (ax1, ax2, ax3), test_State, ds_Landmark_GroundTruth)
        test_Landmark = [landmark for landmark in ds_Landmark_GroundTruth if landmark.id in test_LM_id]
        plot_landmarks(fig, ax1, [test_Landmark[0]])
        plot_landmarks(fig, ax2, [test_Landmark[1]])
        plot_landmarks(fig, ax3, [test_Landmark[2]])

    # --- Part B --- 
    
    # Q7:
    if q7:
        # need to synchronize data timestamps
        dt = 1/50
        ds_Control, ds_GroundTruth, ds_Landmark_GroundTruth, ds_Measurement = import_data(i, resample=True, dt=dt)

        #UKF parameters
        state_0 = ds_GroundTruth[0] # initial state from ground truth
        alpha, beta, kappa = 0.001, 2, 0
        state_0.P = np.diag([0.01, 0.01, 0.01]) # initial covariance
        Q = np.diag([0.01, 0.01, 0.001]) # process noise covariance
        R = np.diag([0.01, 0.01]) # measurement noise covariance

        UKF_State = []
        UKF_State.append(state_0)
        for control in ds_Control: # [0:2000]
            prior = UKF_State[-1]
            measurements = [measurement for measurement in ds_Measurement if abs(measurement.t - control.t) < 1e-5] # find all measurements at this time step
            posterior = ukf(prior, control, measurements, ds_Landmark_GroundTruth, alpha, beta, kappa, Q, R) # accounts for no measurements if list is empty

            UKF_State.append(posterior)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
        fig.suptitle(f"Q7: Robot trajectories, UKF, based on ds{i}_Control.dat and ds{i}_Measurement.dat")
        plot_state(fig, ax1, UKF_State, "UKF")
        plot_state(fig, ax1, ds_GroundTruth, "Ground truth")
        plot_landmarks(fig, ax1, ds_Landmark_GroundTruth)

        ax2.set_title(f"Zoomed in view")
        plot_state(fig, ax2, UKF_State, "Motion model")
        plot_state(fig, ax2, ds_GroundTruth, "Ground truth")
        plot_landmarks(fig, ax2, ds_Landmark_GroundTruth)
        zoom = 1
        ax2.set_xlim(ds_GroundTruth[0].x[0]-zoom, ds_GroundTruth[0].x[0]+zoom)
        ax2.set_ylim(ds_GroundTruth[0].x[1]-zoom, ds_GroundTruth[0].x[1]+zoom)


    plt.show()

### DATA IMPORT ###

def import_data(i, resample=False, dt=0.1, robot_id=False):
    ds_Control_raw = import_dat(f'ds{i}/ds{i}_Control.dat')
    ds_GroundTruth_raw = import_dat(f'ds{i}/ds{i}_GroundTruth.dat')
    ds_Landmark_GroundTruth_raw = import_dat(f'ds{i}/ds{i}_Landmark_GroundTruth.dat') 
    ds_Measurement_raw = import_dat(f'ds{i}/ds{i}_Measurement.dat') 
    ds_Barcodes = import_dat(f'ds{i}/ds{i}_Barcodes.dat')  

    if resample:
        ds_Control_raw, ds_Measurement_raw, ds_GroundTruth_raw = resample_data(dt, ds_Control_raw, ds_Measurement_raw, ds_GroundTruth_raw)
        ds_Control = [Control(control[0], control[1], control[2], dt) for control in ds_Control_raw]
    else:
        ds_Control = []
        for j, control in enumerate(ds_Control_raw): # convert into list of Control objects
            dt =  ds_Control_raw[j+1][0] - control[0] if j < len(ds_Control_raw) - 1 else control[0] - ds_Control_raw[j-1][0] # compute dt, exception for last element
            control = Control(control[0], control[1], control[2], dt)
            ds_Control.append(control)
    
    ds_Measurement = [Measurement(row[0], row[1], row[2], row[3]) for row in ds_Measurement_raw] # convert into list of Measurement objects
    ds_GroundTruth = [State(t=row[0], x=[row[1], row[2], row[3]]) for row in ds_GroundTruth_raw] # convert into list of State objects
    ds_Landmark_GroundTruth = [Landmark(row[0], [row[1], row[2]], [row[3], row[4]]) for row in ds_Landmark_GroundTruth_raw] # convert into list of Landmark objects
    
    for measurement in ds_Measurement:
        for j in range(len(ds_Barcodes)):
            if ds_Barcodes[j][1] == measurement.id:
                measurement.id = int(ds_Barcodes[j][0]) # replace barcode with id
                break

    if robot_id:
        robot_id = get_robot_id(ds_Measurement, ds_Barcodes)

    return ds_Control, ds_GroundTruth, ds_Landmark_GroundTruth, ds_Measurement

def import_dat(filename):
    with open(filename, 'r') as file:
        return np.loadtxt(file)
    
def resample_data(dt, ds_Control_raw, ds_Measurement_raw, ds_GroundTruth_raw):
    # Control and GroundTruth linearly interpolated to fixed timestep
    # measurements rounded to nearest timestep

    min_time = ds_GroundTruth_raw[0, 0]
    ds_Control_raw[:, 0] -= min_time
    ds_GroundTruth_raw[:, 0] -= min_time
    ds_Measurement_raw[:, 0] -= min_time

    max_time = ds_GroundTruth_raw[-1, 0]
    timesteps = int(max_time / dt) + 1
    t_grid = np.linspace(0, max_time, timesteps)

    ds_Control = np.zeros((timesteps, ds_Control_raw.shape[1]))
    ds_Control[:, 0] = t_grid
    for i in range(1, ds_Control_raw.shape[1]):
        ds_Control[:, i] = np.interp(t_grid, ds_Control_raw[:, 0], ds_Control_raw[:, i])

    ds_GroundTruth = np.zeros((timesteps, ds_GroundTruth_raw.shape[1]))
    ds_GroundTruth[:, 0] = t_grid
    for i in range(1, ds_GroundTruth_raw.shape[1]):
        ds_GroundTruth[:, i] = np.interp(t_grid, ds_GroundTruth_raw[:, 0], ds_GroundTruth_raw[:, i])

    ds_Measurement = ds_Measurement_raw.copy()
    ds_Measurement[:, 0] = np.round(ds_Measurement[:, 0] / dt) * dt
    ds_Measurement = ds_Measurement[ds_Measurement[:, 0] <= max_time]

    return ds_Control, ds_Measurement, ds_GroundTruth
    
def get_robot_id(ds_Measurement, ds_Barcodes): # find robot id
    ids = set()
    ids.update(measurement.id for measurement in ds_Measurement)
    robot_id_possible = set()
    robot_id_possible.add([int(i) for i in ds_Barcodes[:, 0] if i not in ids][0]) # find the robot id in the barcodes
    if len(robot_id_possible) != 1:
        print(f"More than one possible robot: {robot_id_possible}")
        return None
    else:
        robot_id = int(robot_id_possible.pop())
        print(f"Robot id is {robot_id}")
        return robot_id

### MOTION MODEL ###

def motion_model(prior: "State", control: "Control"): # uses prior state and a control object, returns the next state
    x = np.zeros(prior.x.shape)
    if control.omega == 0:
        x[0] = prior.x[0] + control.v * math.cos(prior.x[2]) * control.dt
        x[1] = prior.x[1] + control.v * math.sin(prior.x[2]) * control.dt
        x[2] = prior.x[2]
    else:
        x[0] = prior.x[0] + (control.v / control.omega) * (math.sin(prior.x[2] + control.omega * control.dt) - math.sin(prior.x[2]))
        x[1] = prior.x[1] + (control.v / control.omega) * (math.cos(prior.x[2]) - math.cos(prior.x[2] + control.omega * control.dt))
        x[2] = prior.x[2] + control.omega * control.dt
    posterior = State(control.t, x=x)
    return posterior

def dead_reckoning(state_0: "State", ds_Control: list["Control"]): # uses initial state and a list of control objects, returns a list of state objects
    DR_State = []
    DR_State.append(state_0)
    for control in ds_Control:
        prior = DR_State[-1]
        posterior = motion_model(prior, control)
        DR_State.append(posterior)
    return DR_State

### MEASUREMENT MODEL ###

def measurement_model(state: "State", ds_Landmark_GroundTruth): # uses current state and landmark ground truth as input, returns estimated landmarks as output
    z_est = [] # list of estimated measurements as Measurement objects
    for landmark in ds_Landmark_GroundTruth:
        range = math.sqrt((landmark.x[0] - state.x[0])**2 + (landmark.x[1] - state.x[1])**2)
        # stddev_range = math.sqrt(landmark.stddev[0]**2 + landmark.stddev[1]**2) # not accounting for robot position uncertainty
        bearing = math.atan2(landmark.x[1] - state.x[1], landmark.x[0] - state.x[0]) - state.x[2]
        z_est.append(Measurement(state.t, landmark.id, range, bearing))
    return z_est

def get_xy_measurement(state: "State", measurement: "Measurement"):
    x = state.x[0] + measurement.range * math.cos(measurement.bearing + state.x[2])
    y = state.x[1] + measurement.range * math.sin(measurement.bearing + state.x[2])
    return x, y

### UKF ###

def ukf(prior: "State", control: "Control", measurements: list["Measurement"], ds_Landmark_GroundTruth: list["Landmark"], alpha, beta, kappa, Q, R):
    # Prediction step
    X_np = generate_sigma_points(prior, alpha, beta, kappa) # X are the sigma points, numpy array
    Y = [motion_model(State(x=sp), control) for sp in X_np] # Y are the propagated sigma points
    Y_np = np.array([sp.x for sp in Y]) # convert object to numpy array
    y_mean, Pyy = compute_mean_and_covariance(Y_np, Q, alpha, beta, kappa)

    measurements = []
    if len(measurements) == 0: # no measurements, return prediction as posterior
        posterior = State(control.t, x=y_mean, P=Pyy)
    else: # Correction step
        measurement = measurements[0] # use the first measurement only for correction step
        Z = [measurement_model(sp, ds_Landmark_GroundTruth) for sp in Y] # Z are the measurement sigma points
        Z_np = np.array([[m.range, m.bearing] for z in Z for m in z if m.id == measurement.id]) # convert object to numpy array, only use the measurement with the same id as the actual measurement
        z_mean, Pzz = compute_mean_and_covariance(Z_np, R, alpha, beta, kappa)

        Pyz = compute_cross_covariance(Y, y_mean, Z, z_mean, alpha, beta, kappa) # cross covariance
        K = Pyz @ np.linalg.inv(Pzz) # Kalman gain

        measurement_ = np.array([measurement.bearing, measurement.range])
        innovation = measurement_ - z_mean # measurement innovation
        
        x = y_mean + K @ innovation
        P = Pyy - K @ Pzz @ K.T
        posterior = State(control.t, x=x, P=P)

    return posterior

def generate_sigma_points(state: "State", alpha, beta, kappa): # need to convert state to numpy array
    n = state.x.shape[0]
    lambda_ = alpha**2 * (n + kappa) - n
    sigma_points = np.zeros((2 * n + 1, 3))
    sigma_points[0] = state.x
    sqrt_matrix = np.linalg.cholesky((n + lambda_) * state.P)
    for i in range(n):
        sigma_points[i + 1] = state.x + sqrt_matrix[:, i]
        sigma_points[i + 1 + n] = state.x - sqrt_matrix[:, i]
    return sigma_points

def compute_mean_and_covariance(Y, Q, alpha, beta, kappa): # (Y, Q) or (Z, R)
    n = Y.shape[1]
    weights_mean = np.zeros(2 * n + 1)
    weights_cov = np.zeros(2 * n + 1)
    lambda_ = alpha**2 * (n + kappa) - n
    weights_mean[0] = lambda_ / (n + lambda_)
    weights_cov[0] = lambda_ / (n + lambda_) + (1 - alpha**2 + beta)
    for i in range(1, 2 * n + 1):
        weights_mean[i] = 1 / (2 * (n + lambda_))
        weights_cov[i] = 1 / (2 * (n + lambda_))

    mean = weights_mean @ Y
    cov = ((Y - mean) * weights_cov[:,None]).T @ (Y - mean) + Q
    return mean, cov

def compute_cross_covariance(Y, y_mean, Z, z_mean, alpha, beta, kappa):
    n = Y.shape[1]
    m = Z.shape[1]
    weights_cov = np.zeros(2 * n + 1)
    lambda_ = alpha**2 * (n + kappa) - n
    weights_cov[0] = lambda_ / (n + lambda_) + (1 - alpha**2 + beta)
    for i in range(1, 2 * n + 1):
        weights_cov[i] = 1 / (2 * (n + lambda_))
    cross_cov =  ((Y - y_mean) * weights_cov[:,None]).T @ (Z - z_mean)
    return cross_cov

### PLOTTING ###

def plot_state(fig, ax, ds_State, label):
    ds_x = [state.x[0] for state in ds_State]
    ds_y = [state.x[1] for state in ds_State]
    ds_theta = [state.x[2] for state in ds_State]

    ax.scatter(ds_x[0], ds_y[0], marker='x', label=f'Start {label}')
    ax.scatter(ds_x[-1], ds_y[-1], marker='*', label=f'End {label}')

    ax.plot(ds_x, ds_y, label=label)
    # ax.quiver(x, y, np.cos(theta), np.sin(theta), color='r', scale=20)
    ax.quiver(ds_x[0], ds_y[0], math.cos(ds_theta[0]), math.sin(ds_theta[0]), width=0.005, scale=20, color='r', label=f'Start orientation {label}')
    ax.quiver(ds_x[-1], ds_y[-1], math.cos(ds_theta[-1]), math.sin(ds_theta[-1]), width=0.005, scale=20, color='b', label=f'End orientation {label}')
    ax.set_xlabel('X position (m)')
    ax.set_ylabel('Y position (m)')
    ax.legend()
    ax.axis('equal')
    ax.grid(True)

def plot_landmarks(fig, ax, ds_Landmarks):
    x = [landmark.x[0] for landmark in ds_Landmarks]
    y = [landmark.x[1] for landmark in ds_Landmarks]
    ax.scatter(x, y, marker='o', color='black', label='Landmarks')
    for landmark in ds_Landmarks:
        ax.text(landmark.x[0], landmark.x[1], f'LM{landmark.id}')
        plot_uncertainty_ellipse(ax, landmark)

def plot_uncertainty_ellipse(ax, landmark): # plot uncertainty ellipse for landmark, 2 times standard deviation
    from matplotlib.patches import Ellipse
    ellipse = Ellipse(xy=(landmark.x[0], landmark.x[1]),
                        width=2*landmark.stddev[0], height=2*landmark.stddev[1],
                        edgecolor='blue', fc='None', lw=1, label='Uncertainty')
    ax.add_patch(ellipse)

def plot_measurement_predictions(fig, axes, ds_State, ds_Landmark_GroundTruth):
    for i, state in enumerate(ds_State):
        ax = axes[i]
        ax.set_title(f"(x, y, theta) = ({state.x[0]:.2f} m, {state.x[1]:.2f} m, {state.x[2]:.2f} rad)")
        for landmark in state.LM_est:
            ax.scatter(landmark.x[0], landmark.x[1], marker='x', color='red')
            ax.text(landmark.x[0], landmark.x[1], f'LM prediction {landmark.id}')
            for landmark_gt in ds_Landmark_GroundTruth:
                if landmark.id == landmark_gt.id:
                    ax.scatter(landmark_gt.x[0], landmark_gt.x[1], marker='o', color='black')
                    ax.text(landmark_gt.x[0], landmark_gt.x[1], f'LM ground truth {landmark_gt.id}')
        ax.scatter(state.x[0], state.x[1], marker='*', color='green', label='Robot position')
        ax.quiver(state.x[0], state.x[1], math.cos(state.x[2]), math.sin(state.x[2]), width=0.005, scale=20, color='green', label='Robot orientation')
        ax.set_xlabel('X position (m)')
        ax.set_ylabel('Y position (m)')
        ax.legend()
        ax.axis('equal')
        ax.grid(True)


### DATA STRUCTURES ###

class State:
    def __init__(self, t=0.0, x=[0.0, 0.0, 0.0], P=np.zeros((3, 3))):
        self.t = t
        self.x = np.array(x) # state (x, y, theta) at t (posterior)
        self.P = P # covariance matrix at t

class Control:
    def __init__(self, t, v, omega, dt):
        self.t = t
        self.v = v
        self.omega = omega
        self.dt = dt

class Landmark:
    def __init__(self, id, x=[0.0, 0.0], stddev=[0.0, 0.0]):
        self.id = int(id)
        self.x = np.array(x) # position (x, y)
        self.stddev = np.array(stddev) # standard deviation (x, y)

class Measurement: # used for measurements and estimated measurements
    def __init__(self, t, id, range, bearing):
        self.t = t
        self.id = int(id)
        self.range = range
        self.bearing = bearing

if __name__ == "__main__":
    main()