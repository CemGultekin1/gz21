## Visualize Arakawa C-grid numbering
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['mathtext.rm'] = 'serif'


nx = 3
ny = 3

Lx = nx
Ly = ny

dx = Lx/nx
dy = Ly/ny

# grid vectors for T-points
x_T = np.arange(dx/2.,Lx,dx)
y_T = np.arange(dy/2.,Ly,dy)

# grid vectors for u-points
x_u = x_T[:-1] + dx/2.
y_u = y_T

#grid vectors for v-points
x_v = x_T
y_v = y_T[:-1] + dy/2.

# grid vectors for q-points
x_q = np.arange(0,Lx+dx/2.,dx)
y_q = np.arange(0,Ly+dy/2.,dy)

## meshgrid

xx_T,yy_T = np.meshgrid(x_T,y_T)
xx_u,yy_u = np.meshgrid(x_u,y_u)
xx_v,yy_v = np.meshgrid(x_v,y_v)
xx_q,yy_q = np.meshgrid(x_q,y_q)

xx_T = xx_T.flatten()
xx_u = xx_u.flatten()
xx_v = xx_v.flatten()
xx_q = xx_q.flatten()

yy_T = yy_T.flatten()
yy_u = yy_u.flatten()
yy_v = yy_v.flatten()
yy_q = yy_q.flatten()

fig,ax = plt.subplots(1,1,figsize=(3,3))

ax.plot(xx_T,yy_T,'r.',ms=10,mec='k')
ax.plot(xx_u,yy_u,'g.',ms=10,mec='k')
ax.plot(xx_v,yy_v,'b.',ms=10,mec='k')
ax.plot(xx_q,yy_q,'k.',ms=10)

ax.plot(x_v,[0]*nx,'b.',ms=10,alpha=.6)
ax.plot(x_v,[Ly]*nx,'b.',ms=10,alpha=.6)
ax.plot([0]*ny,y_u,'g.',ms=10,alpha=.6)
ax.plot([Lx]*ny,y_u,'g.',ms=10,alpha=.6)


plt.grid()

#for xx,yy,vv in zip([xx_T,xx_u,xx_v,xx_q],[yy_T,yy_u,yy_v,yy_q],['T','u','v','q']):
#    for i in range(len(xx)):
#        ax.text(xx[i]+0.03,yy[i]+0.03,r'$'+vv+'_{%i}$' %(i+1),fontsize=15)

s = 0.02

ax.text(xx_T[4]+s,yy_T[4]+s,r'$T$',fontsize=15)
ax.text(xx_u[3]+s,yy_u[3]+s,r'$u^\rightarrow$',fontsize=15)
ax.text(xx_u[2]+s,yy_u[2]+s,r'$u^\leftarrow$',fontsize=15)

ax.text(xx_v[4]+s,yy_v[4]+s,r'$v^\uparrow$',fontsize=15)
ax.text(xx_v[1]+s,yy_v[1]+s,r'$v^\downarrow$',fontsize=15)

ax.text(xx_q[5]+s,yy_q[5]+s,r'$q^\swarrow$',fontsize=15)
ax.text(xx_q[6]+s,yy_q[6]+s,r'$q^\searrow$',fontsize=15)

ax.text(xx_q[9]+s,yy_q[9]+s,r'$q^\nwarrow$',fontsize=15)
ax.text(xx_q[10]+s,yy_q[10]+s,r'$q^\nearrow$',fontsize=15)

ax.set_xlabel(r'$\Delta x$',size=15)
ax.set_ylabel(r'$\Delta y$',size=15)

ax.set_xticks(range(Lx+1))
ax.set_yticks(range(Ly+1))

ax.set_xlim(0.8,2.2)
ax.set_ylim(0.8,2.2)

ax.set_frame_on(False)

#dxstring = [(r'$%i\Delta x$' % i) if i > 1 else r'$\Delta x$' for i in range(1,nx)]
#dystring = [(r'$%i\Delta y$' % i) if i > 1 else r'$\Delta y$' for i in range(1,ny)]

#ax.set_xticklabels([r'$0$']+dxstring+[r'$L_x$'])
#ax.set_yticklabels([r'$0$']+dystring+[r'$L_y$'])
ax.set_xticklabels([])
ax.set_yticklabels([])
plt.tight_layout()
fig.savefig(path+'matvis/relative_nodes.png',dpi=150)
plt.close(fig)