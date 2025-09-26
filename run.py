import numpy as np
import math
import matplotlib.pyplot as plt

def main():
    i = 0 # dataset index
    partA = True
    partB = False

    export = False # export resampled data files
    use_resampled = False # use resampled data files
    dt = 1/10 # resampling timestep

    if use_resampled:
        dt = None
        export = False
        i = str(i) + "_RS"
        
    ds_Control, ds_GroundTruth, ds_Landmark_GroundTruth, ds_Measurement = import_data(i, dt=dt, export=export)

    # ------------- Part A -------------
    if partA:
        # Q2:
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
        ax.set_title("Robot trajectories, dead reckoning, based on the 6 given control inputs")
        plot_state(fig, ax, DR_State, "Motion model")

        # Q3:
        DR_State = dead_reckoning(ds_GroundTruth[0], ds_Control) # starting at first ground truth state

        title = f"Robot trajectories, dead reckoning, based on ds{i}_Control.dat and ds{i}_Measurement.dat"
        ds_ = [[ds_GroundTruth, DR_State], ds_Landmark_GroundTruth]
        labels = ["Ground truth", "Dead reckoning"]
        colors = ["blue", "orange"]
        ds_Plot(ds_, title, labels, colors)

        # Q6:
        test_State = [State(x=[2.0, 3.0, 0.0]), State(x=[0.0, 3.0, 0.0]), State(x=[1.0, -2.0, 0.0])]
        test_LM_id = [6, 13, 17]

        for j, state in enumerate(test_State):
            landmark = [landmark_ for landmark_ in ds_Landmark_GroundTruth if landmark_.id == test_LM_id[j]][0] # extract single landmark from list
            measurement = measurement_model(state, landmark)
            x, y = get_xy_measurement(state, measurement)
            print(f"\nRobot position: \n(x, y, theta) = ({state.x[0]:.3f} m, {state.x[1]:.3f} m, {state.x[2]:.3f} rad)")
            print(f"Landmark {measurement.id} predicted at: \n(range, bearing) = ({measurement.z[0]:.3f} m, {measurement.z[1]:.3f} rad)")
            print(f"(x, y) = ({x:.3f} m, {y:.3f} m)")
            print(f"Landmark {landmark.id} ground truth at: \n(x, y) = ({landmark.x[0]:.3f} m, {landmark.x[1]:.3f} m)")
            break

    # ------------- Part B -------------
    if partB:
        #UKF parameters
        state_0 = ds_GroundTruth[0] # initial state from ground truth
        state_0.P = np.diag([0.001, 0.001, 0.0001]) # initial covariance
        Q = np.diag([0.001, 0.001, 0.0001]) # process noise covariance
        R = np.diag([0.001, 0.0001]) # measurement noise covariance
        alpha, kappa, beta = 0.3, 0.0, 2.0
        weights_mean, weights_cov = compute_weights(3, alpha, kappa, beta) # n=3 for (x, y, theta)

        UKF_State = []
        UKF_State.append(state_0)
        for control in ds_Control: # first measurement happens at t=11.12, step 557
            prior = UKF_State[-1]
            measurements = [measurement for measurement in ds_Measurement if abs(measurement.t - control.t) < 1e-5] # find all measurements at this time step
            posterior = ukf(prior, control, measurements, ds_Landmark_GroundTruth, Q, R, weights_mean, weights_cov, alpha, kappa, beta) # accounts for no measurements if list is empty
            UKF_State.append(posterior)

        print(f"\n Final state ground truth / estimate")
        print(f"(x, y, theta) = ({ds_GroundTruth[-1].x[0]:.3f} m, {ds_GroundTruth[-1].x[1]:.3f} m, {ds_GroundTruth[-1].x[2]:.3f} rad)")
        print(f"(x, y, theta) = ({UKF_State[-1].x[0]:.3f} m, {UKF_State[-1].x[1]:.3f} m, {UKF_State[-1].x[2]:.3f} rad)")

        DR_State = dead_reckoning(ds_GroundTruth[0], ds_Control) # as a reference for now

        title = f"Robot trajectories, UKF, based on ds{i}_Control.dat and ds{i}_Measurement.dat"
        ds_ = [[ds_GroundTruth, DR_State, UKF_State], ds_Landmark_GroundTruth]
        labels = ["Ground truth", "Dead reckoning", "UKF"]
        colors = ["blue", "orange", "green"]
        ds_Plot(ds_, title, labels, colors)


    plt.show()

### DATA IMPORT ###

def import_data(i, dt=None, export=False, robot_id=False):
    ds_Control_raw = import_dat(f'ds{i}/ds{i}_Control.dat')
    ds_GroundTruth_raw = import_dat(f'ds{i}/ds{i}_GroundTruth.dat')
    ds_Landmark_GroundTruth_raw = import_dat(f'ds{i}/ds{i}_Landmark_GroundTruth.dat') 
    ds_Measurement_raw = import_dat(f'ds{i}/ds{i}_Measurement.dat') 
    ds_Barcodes = import_dat(f'ds{i}/ds{i}_Barcodes.dat')  

    if dt is not None:
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

    if export == True and dt is not None: # export resampled data files
        i = str(i) + "_RS" # resampled
        fn_control = f'ds{i}/ds{i}_Control.dat'
        fn_measurement = f'ds{i}/ds{i}_Measurement.dat'
        fn_groundtruth = f'ds{i}/ds{i}_GroundTruth.dat'
        export_dat(fn_control, ds_Control_raw)
        export_dat(fn_measurement, ds_Measurement_raw)
        export_dat(fn_groundtruth, ds_GroundTruth_raw)

        fn_barecodes = f'ds{i}/ds{i}_Barcodes.dat' # adding to have complete ds_RS folder
        fn_landmark = f'ds{i}/ds{i}_Landmark_GroundTruth.dat'
        export_dat(fn_barecodes, ds_Barcodes)
        export_dat(fn_landmark, ds_Landmark_GroundTruth_raw)


    return ds_Control, ds_GroundTruth, ds_Landmark_GroundTruth, ds_Measurement

def import_dat(filename):
    with open(filename, 'r') as file:
        return np.loadtxt(file)

def export_dat(filename, data):
    with open(f'{filename}', 'w') as file:
        np.savetxt(file, data, fmt='%.3f')

def resample_data(dt, ds_Control_raw, ds_Measurement_raw, ds_GroundTruth_raw):
    # Control and GroundTruth linearly interpolated to fixed timestep
    # Measurements rounded to nearest timestep

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

def motion_model(prior: "State", control: "Control", Q=np.zeros((3, 3))): # uses prior state and a control object, returns the next state
    x = np.zeros(prior.x.shape)
    if control.omega == 0:
        x[0] = prior.x[0] + control.v * math.cos(prior.x[2]) * control.dt #+ np.random.normal(0, math.sqrt(Q[0,0]))
        x[1] = prior.x[1] + control.v * math.sin(prior.x[2]) * control.dt #+ np.random.normal(0, math.sqrt(Q[1,1]))
        x[2] = prior.x[2] #+ np.random.normal(0, math.sqrt(Q[2,2]))
    else:
        x[0] = prior.x[0] + (control.v / control.omega) * (math.sin(prior.x[2] + control.omega * control.dt) - math.sin(prior.x[2])) #+ np.random.normal(0, math.sqrt(Q[0,0]))
        x[1] = prior.x[1] + (control.v / control.omega) * (math.cos(prior.x[2]) - math.cos(prior.x[2] + control.omega * control.dt)) #+ np.random.normal(0, math.sqrt(Q[1,1]))
        x[2] = prior.x[2] + control.omega * control.dt #+ np.random.normal(0, math.sqrt(Q[2,2]))
    x[-1] = normalize_angle(x[-1])
    posterior = State(control.t, x=x)
    return posterior

def dead_reckoning(state_0: "State", ds_Control: list["Control"]): # loops motion model
    DR_State = []
    DR_State.append(state_0)
    for control in ds_Control:
        prior = DR_State[-1]
        posterior = motion_model(prior, control)
        DR_State.append(posterior)
    return DR_State

def normalize_angle(angle): # shift to ]-pi, pi]
    return (angle + math.pi) % (2 * math.pi) - math.pi 

### MEASUREMENT MODEL ###

def measurement_model(state: "State", landmark: "Landmark", R=np.zeros((2, 2))): # uses current state and landmark ground truth as input, returns estimated landmarks as output
    range = math.sqrt((landmark.x[0] - state.x[0])**2 + (landmark.x[1] - state.x[1])**2) #+ np.random.normal(0, math.sqrt(R[0,0]))
    bearing = math.atan2(landmark.x[1] - state.x[1], landmark.x[0] - state.x[0]) - state.x[2] #+ np.random.normal(0, math.sqrt(R[1,1]))
    bearing = normalize_angle(bearing)
    measurement = Measurement(state.t, landmark.id, range, bearing)
    return measurement

def get_xy_measurement(state: "State", measurement: "Measurement"):
    x = state.x[0] + measurement.z[0] * math.cos(measurement.z[1] + state.x[2])
    y = state.x[1] + measurement.z[0] * math.sin(measurement.z[1] + state.x[2])
    return x, y

### UKF ###

def ukf(prior: "State", control: "Control", measurements: list["Measurement"], ds_Landmark_GroundTruth: list["Landmark"], Q, R, weights_mean, weights_cov, alpha, kappa, beta):
    # Prediction step
    X_np = generate_sigma_points(prior, alpha, kappa, beta) # X are the sigma points, numpy array
    Y = [motion_model(State(x=sp), control, Q) for sp in X_np] # Y are the propagated sigma points
    Y_np = np.array([sp.x for sp in Y]) # convert object to numpy array
    y_mean, Pyy = compute_mean_and_covariance(Y_np, Q, weights_mean, weights_cov) # same weights for mean and covariance

    posterior = None
    # measurements = []
    if len(measurements) == 0: # no measurements, return prediction as posterior
        posterior = State(control.t, x=y_mean, P=Pyy)
    else: # Correction step
        for measurement in measurements:
            Y_np = generate_sigma_points(State(x=y_mean, P=Pyy), alpha, kappa, beta) # new sigma points around predicted mean
            Y = [State(x=sp, P=Pyy) for sp in Y_np]
        # measurement = measurements[0] # use the first measurement only for now
            if measurement.id in [landmark_.id for landmark_ in ds_Landmark_GroundTruth]: # if measurement id not in landmark ground truth, return prediction as posterior
                landmark = [landmark_ for landmark_ in ds_Landmark_GroundTruth if landmark_.id == measurement.id][0] #  landmark corresponding to measurement
                Z = [measurement_model(sp, landmark, R) for sp in Y] # Z are the estimated measurement sigma points (list of measurementobjects)
                Z_np = np.array([m_est.z for m_est in Z]) # extract the measurement arrays from the Measurement objects
                z_mean, Pzz = compute_mean_and_covariance(Z_np, R, weights_mean, weights_cov) # same weights for mean and covariance

                Pyz = compute_cross_covariance(Y_np, y_mean, Z_np, z_mean, weights_cov) # cross covariance
                K = Pyz @ np.linalg.inv(Pzz) # Kalman gain
                innovation = measurement.z - z_mean # measurement innovation
                innovation[-1] = normalize_angle(innovation[-1])
                x = y_mean + K @ innovation
                x[-1] = normalize_angle(x[-1])
                P = Pyy - K @ Pzz @ K.T

                posterior = State(control.t, x=x, P=P)

                # for next measurement, use posterior as prior
                y_mean = posterior.x
                Pyy = posterior.P

    if posterior is None:
        posterior = State(control.t, x=y_mean, P=Pyy)

    # motion model error
    motion_model_error = np.array([100 * (posterior.x[i] - y_mean[i]) / y_mean[i] for i in range(len(y_mean))])
    err = np.average(motion_model_error)
    if err > 20.0: # 20% error
        print(f"\n% difference: {[int(m) for m in motion_model_error]}")
        print(f"motion model / posterior:")
        print(y_mean)
        print(posterior.x)

    return posterior

def generate_sigma_points(prior: "State", alpha=1e-3, kappa=0, beta=2): # need to convert state to numpy array
    n = prior.x.shape[0]
    lambda_ = alpha**2 * (n + kappa) - n
    sigma_points = np.zeros((2 * n + 1, n))
    sigma_points[0] = prior.x
    sqrt_matrix = np.linalg.cholesky((n + lambda_) * prior.P)
    for i in range(n):
        sigma_points[i + 1]     = prior.x + sqrt_matrix[:, i]
        sigma_points[i + 1 + n] = prior.x - sqrt_matrix[:, i]
    return sigma_points

def compute_weights(n, alpha, kappa, beta):
    weights_mean = np.zeros(2 * n + 1)
    weights_cov = np.zeros(2 * n + 1)
    lambda_ = alpha**2 * (n + kappa) - n
    weights_mean[0] = lambda_ / (n + lambda_)
    weights_cov[0] = lambda_ / (n + lambda_) + (1 - alpha**2 + beta)
    for i in range(1, 2 * n + 1):
        weights_mean[i] = 1 / (2 * (n + lambda_))
        weights_cov[i] = 1 / (2 * (n + lambda_))
    return weights_mean, weights_cov

def compute_mean_and_covariance(Y, Q, weights_mean, weights_cov): # (Y, Q) or (Z, R)
    mean = np.sum(weights_mean[:, None] * Y, axis=0)
    mean[-1] = math.atan2(np.sum(weights_mean * np.sin(Y[:, -1])), np.sum(weights_mean * np.cos(Y[:, -1]))) # theta, bearing 
    cov = Q.copy()
    for i in range(Y.shape[0]):
        diff = Y[i] - mean
        diff[-1] = normalize_angle(diff[-1])
        cov += weights_cov[i] * np.outer(diff, diff)
    return mean, cov

def compute_cross_covariance(Y, y_mean, Z, z_mean, weights):
    cross_cov = np.zeros((Y.shape[1], Z.shape[1]))
    for i in range(Y.shape[0]):
        dy = Y[i] - y_mean
        dz = Z[i] - z_mean
        dy[-1] = normalize_angle(dy[-1])
        dz[-1] = normalize_angle(dz[-1])
        cross_cov += weights[i] * np.outer(dy, dz)
        # cross_cov += weights[i] * np.outer(Y[i] - y_mean, Z[i] - z_mean)
    return cross_cov

### PLOTTING ###

def ds_Plot(ds_, title, labels, colors):
    ds_State = ds_[0]
    ds_Landmark_GroundTruth = ds_[1]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
    for ax in (ax1, ax2):
        for label, state, color in zip(labels, ds_State, colors):
            plot_state(fig, ax, state, label, color)
        plot_landmarks(fig, ax, ds_Landmark_GroundTruth)
    zoom = 1
    ax2.set_xlim(ds_State[0][0].x[0]-zoom, ds_State[0][0].x[0]+zoom)
    ax2.set_ylim(ds_State[0][0].x[1]-zoom, ds_State[0][0].x[1]+zoom)
    fig.suptitle(title)

def plot_state(fig, ax, ds_State, label, color='blue'):
    ds_x = [state.x[0] for state in ds_State]
    ds_y = [state.x[1] for state in ds_State]
    ds_theta = [state.x[2] for state in ds_State]

    m = 20 # quiver spacing
    ds_x_q = ds_x[::m]
    ds_y_q = ds_y[::m]
    ds_theta_q = ds_theta[::m]

    ax.scatter(ds_x[0], ds_y[0], marker='x', label=f'Start {label}', color=color)
    ax.scatter(ds_x[-1], ds_y[-1], marker='*', label=f'End {label}', color=color)

    ax.plot(ds_x, ds_y, label=label, color=color)
    ax.quiver(ds_x_q, ds_y_q, np.cos(ds_theta_q), np.sin(ds_theta_q), color=color, scale=15, width=0.002, label=f'Orientation {label}', alpha=0.2)
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
    def __init__(self, t=0.0, x=None, P=None):
        self.t = t
        self.x = np.array(x) if x is not None else np.zeros(3) # state (x, y, theta) at t (posterior)
        self.P = P if P is not None else np.zeros((3, 3)) # covariance matrix at t

class Control:
    def __init__(self, t, v, omega, dt):
        self.t = t
        self.v = v
        self.omega = omega
        self.dt = dt

class Landmark:
    def __init__(self, id, x=None, stddev=None):
        self.id = int(id)
        self.x = np.array(x) if x is not None else np.zeros(2) # position (x, y)
        self.stddev = np.array(stddev) if stddev is not None else np.zeros(2) # standard deviation (x, y)

class Measurement: # used for measurements and estimated measurements
    def __init__(self, t, id, range, bearing):
        self.t = t
        self.z = np.array([range, bearing]) # measurement (range, bearing) at t
        self.id = int(id)

if __name__ == "__main__":
    main()
