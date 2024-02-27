import brian2
from brian2.units import *
import numpy as np
import matplotlib.pyplot as plt

np.random.default_rng(12345)
brian2.default_rng(12)

tau = 5*ms

pop_A_ = brian2.PoissonGroup(N=10, rates=10*Hz)

pop_A = brian2.NeuronGroup(
    10,
    'dv/dt = 0/tau : 1',
    threshold='v >= 0.1',
    reset='v=0.0',
    refractory=0*ms,
    method='euler'
)

pop_B = brian2.NeuronGroup(
    10,
    "dv/dt = -v/tau: 1",
    threshold='v > 5',
    reset='v = 0',
    method='exact'
)

pop_A.v = 0.0
pop_B.v = 0

S_ = brian2.Synapses(pop_A_, pop_A,
                     on_pre='v_post = 0.1')
S_.connect(j='i')

S = brian2.Synapses(pop_A, pop_B,
                    model='w:1',
                    on_pre='v += w')

S.connect()
S.w = np.random.normal(2 , 2, size=100)

fig, ax = plt.subplots()
ws = ax.matshow(np.reshape(S.w, (10, 10)))
cbar = fig.colorbar(ws, extend='both')
cbar.minorticks_on()
ax.set_title('Weight Matrix')

monitors = [brian2.SpikeMonitor(pop_A), brian2.SpikeMonitor(pop_B)]

net = brian2.Network(brian2.collect())
net.add(monitors)

net.run(2000*ms)

plt.figure()
brian2.plot(monitors[0].t/ms, monitors[0].i, '.', label='parrots')
brian2.plot(monitors[1].t/ms, monitors[1].i+15, '.', label='pop_B')
plt.title("Spike Times")

plt.legend()


plt.show()