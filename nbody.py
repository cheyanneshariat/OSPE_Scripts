import numpy as np
import matplotlib.pyplot as plt
# import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
from scipy.integrate import odeint

G = 6.67430 * 10**(-11)  # gravitational constant
c = 299792458.0  # speed of light
AU = 1.495978707 * 10**11  # astronomical unit
MSun = 1.98847 * 10**30  # solar mass
yr = 31556952  # year

# function which returns derivative at given time
def func_r_v_dot(r_v,t,N_obj,m): # r_v (positions and velocities) and m (masses) should numpy arrays
    r_v = np.array(r_v) # shape (N_obj*6)
    m = np.array(m)

    r_v_dot = np.zeros((N_obj,6))

    assert r_v.shape == (N_obj*6,)
    assert m.shape == (N_obj,)

    r_v = np.reshape(r_v,(N_obj,6)) # shape N_obj X 6

    for i in range(N_obj):
        r_i = r_v[i,0:3]
        v_i = r_v[i,3:6]
        m_i = m[i]
        a_i = np.zeros(3)
        for j in range(N_obj):
            if i!=j:
                r_j = r_v[j,0:3]
                v_j = r_v[j,3:6]
                m_j = m[j]
                r_ij = r_i-r_j
                r_ij_norm = np.linalg.norm(r_ij)
                a_ij_0 = -G*m_j*r_ij/r_ij_norm**3 # Newtonian acceleration
                a_ij_1 = 0 # 1PN term
                a_ij_25 = 0 # 2.5PN term
                a_i = a_i + a_ij_0 + a_ij_1/c**2 + a_ij_25/c**5
        r_v_dot[i,0:3] = v_i
        r_v_dot[i,3:6] = a_i

    r_v_dot = np.reshape(r_v_dot,(N_obj*6)) # shape (N_obj*6)
    return r_v_dot

# function which returns adaptive time-steps
def func_adapt_dt(r_v,N_obj,m): # r_v (positions and velocities) and m (masses) should numpy arrays
    r_v = np.array(r_v) # shape (N_obj*6)
    m = np.array(m)
    eps = 0

    dt = yr # standard time-step

    assert r_v.shape == (N_obj*6,)
    assert m.shape == (N_obj,)

    r_v = np.reshape(r_v,(N_obj,6)) # shape N_obj X 6

    for i in range(N_obj):
        r_i = r_v[i,0:3]
        v_i = r_v[i,3:6]
        m_i = m[i]
        for j in range(N_obj):
            if i!=j:
                r_j = r_v[j,0:3]
                v_j = r_v[j,3:6]
                m_j = m[j]
                r_ij = r_i-r_j+eps                
                v_ij = v_i-v_j
                r_ij_norm = np.linalg.norm(r_ij)
                v_ij_norm = np.linalg.norm(v_ij)
                a_ij_norm = G*m_j/r_ij_norm**2
                if v_ij_norm == 0 and a_ij_norm != 0:
                    dt_min = np.sqrt(r_ij_norm/a_ij_norm)
                elif v_ij_norm != 0 and a_ij_norm == 0:
                    dt_min = r_ij_norm/v_ij_norm
                elif v_ij_norm != 0 and a_ij_norm != 0:
                    dt_min = min([r_ij_norm/v_ij_norm,np.sqrt(r_ij_norm/a_ij_norm)]) # approximate timescale which needs to be resolved
                else:
                    dt_min = dt
                if dt_min < dt:
                    dt = dt_min
    return dt

# binary
# gifsave = "binary.gif"
# min_xyz,max_xyz = -150,150
# N_obj = 2
# # m = np.ones(N_obj)*10*MSun
# m = np.array([10,30])*MSun
# e_fac = 1.2 # np.sqrt(2) is the limit, 1 is circular
# r_12 = 80*AU
# v_12 = np.sqrt(G*np.sum(m[0:2])/r_12)
# r_v_0 = np.zeros(N_obj*6)
# r_v_0[0:3] = [0,r_12*m[1]/np.sum(m),0]
# r_v_0[3:6] = [e_fac*v_12*m[1]/np.sum(m),0,0]
# r_v_0[6:9] = [0,-r_12*m[0]/np.sum(m),0]
# r_v_0[9:12] = [- e_fac*v_12*m[0]/np.sum(m),0,0]


# triple
gifsave = "triple.gif"
min_xyz,max_xyz = -150,150
N_obj = 3
# m = np.ones(N_obj)*10*MSun
m = np.array([30,10,20])*MSun
e_fac = 1.2 # np.sqrt(2) is the limit, 1 is circular
r_12 = 15*AU
v_12 = np.sqrt(G*np.sum(m[0:2])/r_12)
r_in12_3 = 100*AU
v_in12_3 = np.sqrt(G*np.sum(m)/r_in12_3)
r_v_0 = np.zeros(N_obj*6)
r_v_0[0:3] = [0, r_in12_3*(m[2]/np.sum(m)), r_12*(m[1]/np.sum(m[0:2]))]
r_v_0[3:6] = [e_fac*v_in12_3*(m[2]/np.sum(m)) + e_fac*v_12*(m[1]/np.sum(m[0:2])), 0, 0]
r_v_0[6:9] = [0, r_in12_3*(m[2]/np.sum(m)), - r_12*(m[0]/np.sum(m[0:2]))]
r_v_0[9:12] = [e_fac*v_in12_3*(m[2]/np.sum(m)) - e_fac*v_12*(m[0]/np.sum(m[0:2])), 0, 0]
r_v_0[12:15] = [0, - r_in12_3*(np.sum(m[0:2])/np.sum(m)), 0]
r_v_0[15:18] = [- e_fac*v_in12_3*(np.sum(m[0:2])/np.sum(m)), 0, 0]


# 2+2 quadruple
# gifsave = "2+2_quadruple.gif"
# min_xyz,max_xyz = -150,150
# N_obj = 4
# # m = np.ones(N_obj)*10*MSun
# m = np.array([40,20,30,10])*MSun
# e_fac = 1.2 # np.sqrt(2) is the limit, 1 is circular
# r_12 = 10*AU
# v_12 = np.sqrt(G*np.sum(m[0:2])/r_12)
# r_34 = 10*AU
# v_34 = np.sqrt(G*np.sum(m[2:4])/r_34)
# r_in12_in34 = 100*AU
# v_in12_in34 = np.sqrt(G*np.sum(m)/r_in12_in34)
# r_v_0 = np.zeros(N_obj*6)
# r_v_0[0:3] = [0, r_in12_in34*(np.sum(m[2:4])/np.sum(m)), r_12*(m[1]/np.sum(m[0:2]))]
# r_v_0[3:6] = [e_fac*v_in12_in34*(np.sum(m[2:4])/np.sum(m)) + e_fac*v_12*(m[1]/np.sum(m[0:2])), 0, 0]
# r_v_0[6:9] = [0, r_in12_in34*(np.sum(m[2:4])/np.sum(m)), - r_12*(m[0]/np.sum(m[0:2]))]
# r_v_0[9:12] = [e_fac*v_in12_in34*(np.sum(m[2:4])/np.sum(m)) - e_fac*v_12*(m[0]/np.sum(m[0:2])), 0, 0]
# r_v_0[12:15] = [0, - r_in12_in34*(np.sum(m[0:2])/np.sum(m)), r_34*(m[3]/np.sum(m[2:4]))]
# r_v_0[15:18] = [- e_fac*v_in12_in34*(np.sum(m[0:2])/np.sum(m)) + e_fac*v_34*(m[3]/np.sum(m[2:4])), 0, 0]
# r_v_0[18:21] = [0, - r_in12_in34*(np.sum(m[0:2])/np.sum(m)), - r_34*(m[2]/np.sum(m[2:4]))]
# r_v_0[21:24] = [- e_fac*v_in12_in34*(np.sum(m[0:2])/np.sum(m)) - e_fac*v_34*(m[2]/np.sum(m[2:4])), 0, 0]


# 3+1 quadruple
# gifsave = "3+1_quadruple.gif"
# min_xyz,max_xyz = -500,500
# N_obj = 4
# # m = np.ones(N_obj)*10*MSun
# m = np.array([40,10,30,20])*MSun
# e_fac = 1.2 # np.sqrt(2) is the limit, 1 is circular
# r_12 = 20*AU
# v_12 = np.sqrt(G*np.sum(m[0:2])/r_12)
# r_in12_3 = 80*AU
# v_in12_3 = np.sqrt(G*np.sum(m[0:3])/r_in12_3)
# r_mid123_4 = 350*AU
# v_mid123_4 = np.sqrt(G*np.sum(m)/r_mid123_4)
# r_v_0 = np.zeros(N_obj*6)
# r_v_0[0:3] = [r_in12_3*(m[2]/np.sum(m[0:3])), r_mid123_4*(m[3]/np.sum(m)), r_12*(m[1]/np.sum(m[0:2]))]
# r_v_0[3:6] = [e_fac*v_mid123_4*(m[3]/np.sum(m)) + e_fac*v_12*(m[1]/np.sum(m[0:2])), 0, e_fac*v_in12_3*(m[2]/np.sum(m[0:3]))]
# r_v_0[6:9] = [r_in12_3*(m[2]/np.sum(m[0:3])), r_mid123_4*(m[3]/np.sum(m)), - r_12*(m[0]/np.sum(m[0:2]))]
# r_v_0[9:12] = [e_fac*v_mid123_4*(m[3]/np.sum(m)) - e_fac*v_12*(m[0]/np.sum(m[0:2])), 0, e_fac*v_in12_3*(m[2]/np.sum(m[0:3]))]
# r_v_0[12:15] = [- r_in12_3*(np.sum(m[0:2])/np.sum(m[0:3])), r_mid123_4*(m[3]/np.sum(m)), 0]
# r_v_0[15:18] = [e_fac*v_mid123_4*(m[3]/np.sum(m)), 0, - e_fac*v_in12_3*(np.sum(m[0:2])/np.sum(m[0:3]))]
# r_v_0[18:21] = [0, - r_mid123_4*(np.sum(m[0:3])/np.sum(m)), 0]
# r_v_0[21:24] = [- e_fac*v_mid123_4*(np.sum(m[0:3])/np.sum(m)), 0, 0]

# Pythagorean 3-body
# gifsave = "pythagorean.gif"
# min_xyz,max_xyz = -50,50
# N_obj = 3
# # m = np.ones(N_obj)*10*MSun
# m = np.array([3.0,4.0,5.0])*MSun
# r_v_0 = np.zeros(N_obj*6)
# r_v_0[0:3] = [10.0*AU , 30.0*AU , 0.0*AU]
# r_v_0[3:6] = [0.0 , 0.0 , 0.0]
# r_v_0[6:9] = [-20.0*AU , -10.0*AU , 0.0*AU]
# r_v_0[9:12] = [0.0 , 0.0 , 0.0]
# r_v_0[12:15] = [10.0*AU , -10.0*AU , 0.0*AU]
# r_v_0[15:18] = [0.0 , 0.0 , 0.0]


N_steps = 1000

t = np.zeros(N_steps+1)
r_v_total = np.zeros((N_steps+1,N_obj,6)) # stores positions and velocities after each time-step
r_v_total[0,:,:] = np.reshape(r_v_0,(N_obj,6))

r_v_evolve = r_v_0 # position and velocity at a given evolution time
for step in range(N_steps):
    dt_evolve = func_adapt_dt(r_v_evolve,N_obj,m)
    r_v_evolve = odeint(func_r_v_dot,r_v_evolve,[0,dt_evolve],args=(N_obj,m))[1] # shape 1 X (N_obj*6) -- (0th element is not relevant)

    t[step+1] = t[step]+dt_evolve
    r_v_total[step+1,:,:] = np.reshape(r_v_evolve,(N_obj,6))

print(t[-1]/yr,"yr evolution time")

plt.style.use('dark_background')

fig = plt.figure(figsize=(10,10))
# ax = p3.Axes3D(fig,auto_add_to_figure=False)
# fig.add_axes(ax)
ax = fig.add_subplot(projection='3d')
ax.grid(None)
ax.w_xaxis.pane.fill = False
ax.w_yaxis.pane.fill = False
ax.w_zaxis.pane.fill = False

trajectories = [ax.plot([],[],[],lw=1,alpha=0.7,color='C%d'%(i))[0] for i in range(N_obj)]
particles = [ax.plot([],[],[],'o',markersize=np.sqrt(m[i]/MSun)*2,color='C%d'%(i))[0] for i in range(N_obj)]
ax.set_title('t = %.0f yr'%(t[0]/yr),fontsize=25)

# function to update matplotlib plot for each frame
def animate(t_step,N_obj,r_v,trajectories,particles,t):
    for i in range(N_obj):
        r = r_v[0:t_step+1,i,0:3]/AU # shape * X 3

        ax.set_title('t = %.0f yr'%(t[t_step]/yr),fontsize=25)

        trajectories[i].set_data(r[:,0],r[:,1]) # X and Y axes
        trajectories[i].set_3d_properties(r[:,2]) # Z axis

        particles[i].set_data(r[t_step:t_step+1,0],r[t_step:t_step+1,1]) # X and Y axes
        particles[i].set_3d_properties(r[t_step:t_step+1,2]) # Z axis

# animate N_steps frames with interval 1ms between frames
anim = animation.FuncAnimation(fig, animate, frames=N_steps+1, fargs=(N_obj,r_v_total,trajectories,particles,t), interval=1, blit=False)

# ax.view_init(azim=90, elev=0)
ax.set_xlim3d([min_xyz,max_xyz])
ax.set_xlabel('X [AU]',fontsize=15)
ax.set_ylim3d([min_xyz,max_xyz])
ax.set_ylabel('Y [AU]',fontsize=15)
ax.set_zlim3d([min_xyz,max_xyz])
ax.set_zlabel('Z [AU]',fontsize=15)

plt.show()
anim.save('./gif/%s'%(gifsave))

# CONVERT GIF TO MP4
# ffmpeg -r 50 -i anim.gif anim.mp4