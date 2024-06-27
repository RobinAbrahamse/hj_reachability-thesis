import rosbag
import numpy as np
import matplotlib.pyplot as plt

#### Set up input arrays
mag = 0.7
cycles = 3
vel_strt = 1.5
vel_stop = 2.7
INPUT_DELAY = 255
INPUT_TAIL = 200
input_steer = np.concatenate((99 * [0.], mag*np.sin(np.arange(np.pi/6, np.pi*2*cycles + np.pi/6, (np.pi*2*cycles) / 400))))
input_vel = np.arange(vel_strt, vel_stop, (vel_stop-vel_strt)/len(input_steer))
input_steer = np.concatenate((INPUT_DELAY * [0.], input_steer, INPUT_TAIL * [0.]))
input_vel = np.concatenate((INPUT_DELAY * [0.3], input_vel, INPUT_TAIL * [0.]))

# Input for circular drive
vel_strt = 0.2
vel_stop = 0.3
input_steer = [0.] * 500
# input_steer = np.arange(0.0, 0.2, 0.2/len(input_steer))
input_vel = np.arange(vel_strt, vel_stop, (vel_stop-vel_strt)/len(input_steer))
input_vel = [0.3] * 500
input_steer = np.concatenate((200 * [0.], input_steer, INPUT_TAIL * [0.]))
input_vel = np.concatenate((200 * [0.], input_vel, INPUT_TAIL * [0.]))



#### Extract data from rosbag
bag = rosbag.Bag('bags/2024-03-21-15-36-20.bag')
# print(bag.get_type_and_topic_info())
TIRE_RADIUS = 0.052
TICKS_PER_REVOLUTION = 80
TICK2RAD = 2.0 * np.pi / TICKS_PER_REVOLUTION
TICK2DIST = TICK2RAD * TIRE_RADIUS

def yaw_from_quat(q):
    return np.arctan2(2.0*(q.w*q.z+q.x*q.y), 1.0 - 2.0*(q.y*q.y + q.z*q.z))

 # Right encoder was not reporting at the time, make sure to check
encoder = [((msg.left_ticks + msg.left_ticks) / 2.0 / msg.right_time_delta * 1e6, time.to_time()) for _, msg, time in bag.read_messages(topics=['/lli/encoder'])]
encoder_delta_time = [msg.right_time_delta for _, msg, _ in bag.read_messages(topics=['/lli/encoder'])]
encoder_times = [time for _, time in encoder]
encoder_vel = [point * TICK2DIST for point, _ in encoder]
encoder_w = [point * TICK2RAD for point, _ in encoder]
N = 20
moc_pos_x = [(msg.pose.pose.position.x, time.to_time()) for _, msg, time in bag.read_messages(topics=['/qualisys/svea3/odom'])]
moc_pos_times = [time for _, time in moc_pos_x]
moc_pos_x = [point for point, _ in moc_pos_x]
moc_pos_y = [msg.pose.pose.position.y for _, msg, _ in bag.read_messages(topics=['/qualisys/svea3/odom'])]
moc_pos_yaw = [yaw_from_quat(msg.pose.pose.orientation) for _, msg, _ in bag.read_messages(topics=['/qualisys/svea3/odom'])]
moc_vel_x = [(msg.twist.twist.linear.x, time.to_time()) for _, msg, time in bag.read_messages(topics=['/qualisys/svea3/odom'])]
moc_vel_times = [time for _, time in moc_vel_x]
moc_vel_x = [point for point, _ in moc_vel_x]
moc_vel_y = [msg.twist.twist.linear.y for _, msg, _ in bag.read_messages(topics=['/qualisys/svea3/odom'])]
moc_vel_yaw = [msg.twist.twist.angular.z for _, msg, _ in bag.read_messages(topics=['/qualisys/svea3/odom'])]
bag.close()



#### Calculate moving averages
encoder_vel_ma = np.convolve(encoder_vel, np.ones(N), 'same') / N
encoder_w_ma = np.convolve(encoder_w, np.ones(N), 'same') / N
moc_vel_yaw_ma = np.convolve(moc_vel_yaw, np.ones(N), 'same') / N
# encoder_vel_ma = np.convolve(encoder_vel, np.ones(N), 'same') / N



#### Start timing arrays at 0
t0 = min(min(encoder_times[0], moc_pos_times[0]), moc_vel_times[0])
encoder_times = [t - t0 for t in encoder_times]
moc_pos_times = [t - t0 for t in moc_pos_times]
moc_vel_times = [t - t0 for t in moc_vel_times]
input_times = [0.0106*i for i in range(len(input_vel))]

enc_start_time = [t - sum(encoder_delta_time[:i])/1e6 for i,t in enumerate(encoder_times)]
enc_start_time = np.mean(enc_start_time)

encoder_times_acc = [sum(encoder_delta_time[:i])/1e6 - enc_start_time for i in range(len(encoder_times))]



#### Create ~0.1s data intervals and align measurement data
START_INPUT_DATA = 200
input_state_times = input_times[START_INPUT_DATA::10]
input_state_vel = input_vel[START_INPUT_DATA::10]
input_state_steer = input_steer[START_INPUT_DATA::10]

def closest_value_inc(t, list, idxs, i):
    while idxs[i] + 1 < len(list) and np.abs(list[idxs[i]+1] - t) < np.abs(list[idxs[i]] - t):
        idxs[i] += 1

encoder_state_times = []
encoder_state_vel = []
moc_vel_state_times = []
moc_vel_state_x = []
moc_vel_state_y = []
moc_vel_state_yaw = []
idxs = [0,0]
for i,t in enumerate(input_state_times):
    closest_value_inc(t, encoder_times_acc, idxs, 0)
    encoder_state_times.append(encoder_times_acc[idxs[0]])
    encoder_state_vel.append(encoder_w_ma[idxs[0]])
    closest_value_inc(t, moc_vel_times, idxs, 1)
    moc_vel_state_times.append(moc_vel_times[idxs[1]])
    moc_vel_state_x.append(moc_vel_x[idxs[1]])
    moc_vel_state_y.append(moc_vel_y[idxs[1]])
    moc_vel_state_yaw.append(moc_vel_yaw_ma[idxs[1]])

t_data = np.array(input_state_times.copy()) - input_state_times[0]
y_data = np.array([moc_vel_state_x, moc_vel_state_y, moc_vel_state_yaw, encoder_state_vel])
u_data = np.array([input_state_vel, input_state_steer])



#### Model fitting
import fit_model

# theta_0 = [0.12, 
#            1.0, 
#            6, 
#            8]
C = 80000
theta_0 = [0.12, 
           1.0, 
           C, 
           C]
n = len(t_data)
# sol = fit_model.estimate(y_data, t_data, u_data, theta_0, ([0.05, 0.7, 3000, 2000], [0.2, 1.5, 200000, 200000]))

y0 = [0., 0., 0., 0.]

theta_g = [0.14, 
           0.9, 
           75000, 
           65000]

# theta_g = [0.121, 
#            1.05, 
#            80100, 
#            79900]

print("here")
y_data = fit_model.f_int(y0, t_data, (t_data, u_data), theta_g)
print("starting estimate")
sol = fit_model.estimate(y_data, t_data, u_data, theta_0, ([0.05, 0.7, 3000, 2000], [0.2, 1.5, 200000, 200000]))


# obtain model estimate
# start_idx = 1
# section_len = n
# y0 = [0., 0., 0., 0.]
# model_estimate = fit_model.f_int(y0, t_data, (t_data, u_data), theta_0)


# obtain model estimate over section
# start_idx = 8
# section_len =  n - start_idx
# model_estimate = fit_model.f_int(y_data[:,start_idx], t_data[start_idx:start_idx+section_len+1], (t_data, u_data), theta_0)


# calculate residuals
# out = fit_model.f_resid(y_data, t_data, u_data, theta_0)
# exit()


def f(y, t, u, theta):
    return fit_model.f(y, [t], ([t], np.array([u]).T), theta)

# calculate model start
# y_data[0,0] = 0.3
# y_data[3,0] = 10
# f([ 3.34095946e-04,  1.64275028e-03, -1.16011810e-04,  1.79997123e+01], 0.1, [1.512,0.], theta_0)
# exit()

fig, ax = plt.subplots(nrows=2, ncols=2)
## plot longitudinal velocity
# ax[0][0].plot(t_data, moc_vel_state_x)
ax[0][0].plot(t_data[start_idx:start_idx+section_len], model_estimate[0,:])
ax[0][0].plot(t_data, input_state_vel)
# ax[0][0].set_ylim([-5, 50])

## plot lateral velocity
# ax[0][1].plot(t_data, moc_vel_state_y)
ax[0][1].plot(t_data[start_idx:start_idx+section_len], model_estimate[1,:])
ax[0][1].plot(t_data, input_state_steer)
# ax[0][1].set_ylim([-5, 10])

## plot yaw rate
# ax[1][0].plot(t_data, encoder_state_vel)
ax[1][0].plot(t_data[start_idx:start_idx+section_len], model_estimate[3,:])
ax[1][0].plot(t_data, input_state_vel/TIRE_RADIUS)
# ax[1][0].set_ylim([-5, 10])

## plot wheel angular velocity
# ax[1][1].plot(t_data, moc_vel_state_yaw)
ax[1][1].plot(t_data[start_idx:start_idx+section_len], model_estimate[2,:])
ax[1][1].plot(t_data, input_state_steer)
# ax[1][1].set_ylim([-5, 40])

# legends
# ax[0][0].legend(["x vel", "x vel est", "vel input"])
# ax[0][1].legend(["y vel", "y vel est", "steering input"])
# ax[1][0].legend(["encoder vel", "wheel vel est", "vel input"])
# ax[1][1].legend(["yaw rate", "yaw rate est", "steering input"])
ax[0][0].legend(["x vel", "vel input"])
ax[0][1].legend(["y vel", "steering input"])
ax[1][0].legend(["encoder vel", "vel input"])
ax[1][1].legend(["yaw rate", "steering input"])

plt.show()


###### PLOTS ######
### Plot state measurements
# plt.figure()
# plt.plot(t_data, moc_vel_state_x) #plt.plot(moc_vel_state_times, moc_vel_state_x)
# plt.plot(t_data, moc_vel_state_y) #plt.plot(moc_vel_state_times, moc_vel_state_y)
# plt.plot(t_data, moc_vel_state_yaw) #plt.plot(moc_vel_state_times, moc_vel_state_yaw)
# plt.plot(t_data, encoder_state_vel) #plt.plot(encoder_state_times, encoder_state_vel)
# plt.plot(t_data, input_state_vel) #plt.plot(input_state_times, input_state_vel)
# plt.plot(t_data, input_state_steer) #plt.plot(input_state_times, input_state_steer)
# plt.legend(["x vel", "y vel", "yaw vel", "encoder vel", "input vel", "input steer"])
# plt.show()

### Plot state sample times
# plt.figure()
# plt.plot(input_state_times)
# plt.plot(encoder_state_times)
# plt.plot(moc_vel_state_times)
# plt.show()

### Plot sample time differences
# plt.figure()
# plt.plot([t - input_state_times[i] for i,t in enumerate(input_state_times[1:])])
# plt.plot([t - encoder_state_times[i] for i,t in enumerate(encoder_state_times[1:])])
# plt.plot([t - encoder_times[i] for i,t in enumerate(encoder_times[1:])])
# plt.plot(encoder_delta_time)
# plt.plot([t - moc_pos_times[i] for i,t in enumerate(moc_pos_times[1:])])
# plt.plot([t - moc_vel_times[i] for i,t in enumerate(moc_vel_times[1:])])
# plt.show()


### Plot position over time
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot(moc_pos_x, moc_pos_y, moc_pos_times)
# plt.gca().set_aspect("equal")
# plt.show()

# plt.figure()
# plt.plot(input_times, input_steer)
# plt.plot(moc_vel_times, moc_vel_yaw)
# plt.plot(moc_pos_times, moc_pos_yaw)
# plt.show()


### Plot velocity measurements
# plt.figure()
# plt.plot(moc_vel_times, moc_vel_x)
# plt.plot(input_times, input_vel)
# plt.plot(encoder_times_acc, encoder_vel_ma)
# plt.plot(encoder_state_times, encoder_state_vel)
# plt.show()


### Plot integrated velocity measurements
# moc_vel_dist = []
# enc_vel_dist = []
# for i in range(len(moc_vel_times))[1:]:
#     moc_vel_dist.append(moc_vel_x[i]*(moc_vel_times[i]-moc_vel_times[i-1]))
# for i in range(len(encoder_times))[1:]:
#     enc_vel_dist.append(encoder_vel[i]*(encoder_times[i]-encoder_times[i-1]))

# moc_vel_dist = np.cumsum(moc_vel_dist)
# enc_vel_dist = np.cumsum(enc_vel_dist)

# plt.figure()
# plt.plot(moc_vel_times[1:], moc_vel_dist)
# plt.plot(encoder_times[1:], enc_vel_dist)
# plt.show()
