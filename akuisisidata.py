import nidaqmx
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

# Baca dari DAQ Device
def readdaq():
    task = nidaqmx.Task()
    task.ai_channels.add_ai_voltage_chan("Dev1/ai0", terminal_config=nidaqmx.constants.TerminalConfiguration.RSE, min_val=-5.0, max_val=5.0)
    task.ai_channels.add_ai_voltage_chan("Dev1/ai1", terminal_config=nidaqmx.constants.TerminalConfiguration.RSE, min_val=-5.0, max_val=5.0)
    task.start()
    value = task.read()
    if -0.5 <= value[0] <= 0.5:
        value[0] = 0
    if -0.5 <= value[1] <= 0.5:
        value[1] = 0
    task.stop()
    task.close()
    return value

# Simpan data ke file CSV
def writefiledata(x1, x2):
    save_path = "D:/"
    os.makedirs(save_path, exist_ok=True)
    completeName = os.path.join(save_path, "lirikatas3.csv")
    with open(completeName, "a") as file:
        value1 = str(round(x1, 2))
        value2 = str(round(x2, 2))
        file.write(value1 + "," + value2 + "\n")

# Inisialisasi Logging
Ts = 0.01  # Sampling Time [seconds]
N = 100
k = 1
x_len = N  # Jumlah titik pada grafik
Tmin = -6
Tmax = 6
y_range = [Tmin, Tmax]
data = []

# Buat Figure untuk plotting
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
xs1 = list(range(0, N))
ys1 = [0] * x_len
xs2 = list(range(0, N))
ys2 = [0] * x_len
ax.set_ylim(y_range)

# Garis kosong yang akan diperbarui
line, = ax.plot(xs1, ys1, label='CH1')
line2, = ax.plot(xs2, ys2, label='CH2')

# Konfigurasi plot
plt.title('Real-Time Plot')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude [V]')
plt.legend(loc='lower right', labelspacing=0.25, handlelength=0.4, handletextpad=0.7)
plt.grid()

# Fungsi Logging dan Update Grafik
def logging(i, ys1, ys2):
    global k
    value = readdaq()
    print(str(round(value[0], 1)) + "," + str(round(value[1], 1)))
    data.append(value)
    writefiledata(value[0], value[1])  # Simpan ke file
    time.sleep(Ts)

    k = k + 1

    ys1.append(value[0])
    ys2.append(value[1])

    ys1 = ys1[-x_len:]
    ys2 = ys2[-x_len:]

    line.set_ydata(ys1)
    line2.set_ydata(ys2)
    return line, line2

# Jalankan animasi real-time
ani = animation.FuncAnimation(fig,
                              logging,
                              frames=len(ys1),
                              fargs=(ys1, ys2),
                              interval=1,
                              blit=True)

plt.show()