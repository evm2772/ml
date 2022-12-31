
import numpy as np
from vedo import settings, Line, show

settings.default_font = "Theemim"

# Generate random data
np.random.seed(1)
data = np.random.uniform(0, 1, (25, 100))
X = np.linspace(-1, 1, data.shape[-1])
G = 0.15 * np.exp(-4 * X**2) # use a  gaussian as a weight

# Generate line plots
lines = []
for i, d in enumerate(data):
    pts = np.c_[X, np.zeros_like(X)+i/10, G*d]
    lines.append(Line(pts, lw=3))

# Set up the first frame
axes = dict(xtitle='\Deltat /\mus', ytitle="source", ztitle="")
plt = show(lines, __doc__, axes=axes, elevation=-30, interactive=False, bg='k8')

# vd = Video("anim_lines.mp4")
for i in range(50):
    data[:, 1:] = data[:, :-1]                      # Shift data to the right
    data[:, 0] = np.random.uniform(0, 1, len(data)) # Fill-in new values
    for line, d in zip(lines, data):                    # Update data
        newpts = line.points()
        newpts[:,2] = G * d
        line.points(newpts).cmap('gist_heat_r', newpts[:,2])
    plt.render()
    if plt.escaped: break # if ESC is hit during the loop
    # vd.add_frame()
# vd.close()

plt.interactive().close()


