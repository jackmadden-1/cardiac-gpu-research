import numpy as np
import matplotlib.pyplot as plt


def fitzhugh_nagumo(v, w, stimulus, a=0.7, b=0.8, epsilon=0.08):
    dv_dt = v - (v ** 3) / 3 - w + stimulus
    dw_dt = epsilon * (v + a - b * w)

    return dv_dt, dw_dt


dt = 0.01
total_time = 100.0

times = np.arange(0, total_time, dt)

v = -1.0
w = -0.5

v_history = []
w_history = []

for t in times:

    if 10 <= t <= 11:
        stimulus = 1.0
    else:
        stimulus = 0.0

    dv_dt, dw_dt = fitzhugh_nagumo(v, w, stimulus)

    v = v + dt * dv_dt
    w = w + dt * dw_dt

    v_history.append(v)
    w_history.append(w)


plt.plot(times, v_history, label="v - excitation")
plt.plot(times, w_history, label="w - recovery")

plt.xlabel("Time")
plt.ylabel("State")
plt.title("FitzHugh-Nagumo Single Cell")
plt.legend()

plt.show()