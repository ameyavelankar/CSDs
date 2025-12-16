import numpy as np
import matplotlib.pyplot as plt


def get_last_n_frames_coords(id,df_out_merged,n=10):
    df_play = df_out_merged[df_out_merged['flat_play_id'] == id]
    nfl_ids = df_play['nfl_id'].unique()
    frames = df_play['frame_id'].unique()[-n:]
    num_frames=len(frames)


    #Format is (nfl_id, frame_id, x, y,Role)
    coords = []

    for frame in frames:
        df_frame = df_play[df_play['frame_id'] == frame]

        for _, row in df_frame.iterrows():
            coords.append((row['nfl_id'], row['frame_id'], row['x'], row['y'], row['player_role']))

    return coords

def get_first_n_frames_coords(id,df_out_merged,n=10):
    df_play = df_out_merged[df_out_merged['flat_play_id'] == id]
    nfl_ids = df_play['nfl_id'].unique()
    frames = df_play['frame_id'].unique()[0:n]
    num_frames=len(frames)


    #Format is (nfl_id, frame_id, x, y,Role)
    coords = []

    for frame in frames:
        df_frame = df_play[df_play['frame_id'] == frame]

        for _, row in df_frame.iterrows():
            coords.append((row['nfl_id'], row['frame_id'], row['x'], row['y'], row['player_role']))

    return coords


def plot_multi_point_control_map(
    time_fn_p1,
    time_fn_p2,
    pos1=(0.0, 0.0),
    pos2_list=None,
    sideline_dist=30,
    vel1=None,
    vel2_list=None,
    vel1mag=None,
    vel2mag_list=None,
    combine_fn=None,
    softmax_k=1.0,
    xlim=(-60, 60),
    ylim=(-30, 30),
    resolution=0.5,
    cmap='coolwarm',
    show_time_contours=False,
    show_vel_direc=False,
    show_people=True,
    gauss_Filter=None,
    arrow_len=1.0
):
    """
    Plot a control map where there is one P1 and an arbitrary number of P2 points.

    Args:
      - time_fn_p1: callable f(x_rel, y_rel) -> time for P1 (coords relative to P1).
      - time_fn_p2: callable OR list of callables for P2 (each expects coords relative to that P2).
                    If a single callable is provided it will be used for all P2s.
      - pos1: (x,y) P1 position.
      - pos2_list: list of (x,y) positions for P2 team.
      - vel1: optional P1 velocity; can be an angle (radians) or a (vx,vy) tuple.
      - vel2_list: optional single velocity or list of velocities for P2 players (same format as vel1).
      - vel1mag: optional magnitude for P1 arrow (e.g. speed in m/s).
      - vel2mag_list: optional magnitudes for P2 arrows; can be:
            * None (all default length)
            * scalar (same magnitude for all P2s)
            * list/array per P2
      - combine_fn: callable combine(TT1, TT_others_stack) -> array in [0,1] giving P1 dominance.
                    If None, default softmax across [P1,P2...] is used, returning probability mass for P1.
      - softmax_k: softmax temperature when default combine is used.
      - xlim, ylim, resolution: grid extents and spacing.
      - cmap: matplotlib colormap.
      - show_time_contours: overlay time contours (P1 solid, P2 dashed).
      - arrow_len: arrow length multiplier for velocity arrows.

    Returns:
      (fig, ax, XX, YY, TT1, TT_others_stack, combined)
    """

    if pos2_list is None:
        pos2_list = []

    # grid
    xs = np.arange(xlim[0], xlim[1] + 1e-9, resolution)
    ys = np.arange(ylim[0], ylim[1] + 1e-9, resolution)
    XX, YY = np.meshgrid(xs, ys)  # shape (ny, nx)

    # TT1 (relative to pos1)
    X1 = XX - pos1[0]
    Y1 = YY - pos1[1]
    vec1 = np.vectorize(lambda xr, yr: float(time_fn_p1(xr, yr)))
    TT1 = vec1(X1, Y1).astype(float)

    # TT for each P2
    TT_others = []
    is_list_of_fns = isinstance(time_fn_p2, (list, tuple))
    for i, p2 in enumerate(pos2_list):
        X2 = XX - p2[0]
        Y2 = YY - p2[1]
        if is_list_of_fns:
            fn = time_fn_p2[i]
        else:
            fn = time_fn_p2
        vec = np.vectorize(lambda xr, yr: float(fn(xr, yr)))
        TTj = vec(X2, Y2).astype(float)
        TT_others.append(TTj)
    if TT_others:
        TT_others_stack = np.stack(TT_others, axis=0)  # shape (n2, ny, nx)
    else:
        TT_others_stack = np.zeros((0,) + TT1.shape, dtype=float)

    # default combine: softmax across P1 and P2s -> p1 = weight[P1]
    if combine_fn is None:
        def default_combine(tt1, tt_others, k=softmax_k):
            if tt_others.size == 0:
                return np.ones_like(tt1)
            all_times = np.concatenate([tt1[np.newaxis, ...], tt_others], axis=0)  # (N+1, H, W)
            # stabilize
            mn = np.min(all_times, axis=0)
            A = np.exp(-k * (all_times - mn))
            S = np.sum(A, axis=0) + 1e-12
            return A[0] / S
        combine_fn = default_combine

    combined = combine_fn(TT1, TT_others_stack)
    combined = np.asarray(combined, dtype=float)
    combined = np.nan_to_num(combined, nan=0.0, posinf=1.0, neginf=0.0)
    combined = np.clip(combined, 0.0, 1.0)

    # optional sideline mask
    if sideline_dist is not None:
        defender_mask = YY < -sideline_dist
        combined = np.where(defender_mask, 0.0, combined)

    # optional Gaussian smoothing for visualization only
    if gauss_Filter is not None:
        from scipy.ndimage import gaussian_filter
        combined_plot = gaussian_filter(combined, sigma=gauss_Filter[0], truncate=gauss_Filter[1])
    else:
        combined_plot = combined

    # plotting
    fig, ax = plt.subplots(figsize=(9, 6))
    pcm = ax.pcolormesh(XX, YY, combined_plot, cmap=cmap, shading='auto', vmin=0, vmax=1)

    if show_people:
        # markers
        ax.scatter([pos1[0]], [pos1[1]], c='yellow', edgecolors='k', s=90,
                   zorder=6, label='Ball Carrier')
        if pos2_list:
            p2xs, p2ys = zip(*pos2_list)
            ax.scatter(p2xs, p2ys, c='pink', edgecolors='k', s=70,
                       zorder=6, label='Defenders')
        ax.legend(loc='upper right')

        # helper to draw a velocity arrow: vel can be angle (radians) or (vx,vy)
        def _draw_arrow(p, vel, mag=None, color='k'):
            """
            p   : (x, y) position
            vel : angle in radians OR (vx, vy) direction
            mag : scalar speed; if None, arrow_len alone sets the length
            """
            if vel is None:
                return
            try:
                # interpret vel as direction
                if isinstance(vel, (list, tuple, np.ndarray)) and len(vel) == 2:
                    vx, vy = float(vel[0]), float(vel[1])
                    norm = np.hypot(vx, vy) or 1.0
                    dirx, diry = vx / norm, vy / norm
                else:
                    angle = float(vel)
                    dirx, diry = np.cos(angle), np.sin(angle)

                # choose arrow length
                if mag is None:
                    length = arrow_len
                else:
                    length = arrow_len * float(mag)

                dx = dirx * length
                dy = diry * length

                ax.arrow(
                    p[0], p[1], dx, dy,
                    head_width=0.3, head_length=0.3,
                    fc=color, ec='k', zorder=7
                )
            except Exception:
                pass

        if show_vel_direc:
            # P1 arrow
            _draw_arrow(pos1, vel1, mag=vel1mag, color='yellow')

            # P2 arrows (vel2_list may be None, a single vel, or a list matching pos2_list)
            if pos2_list and vel2_list is not None:

                # helper to get per-defender magnitude
                def _get_mag(i):
                    if vel2mag_list is None:
                        return None
                    # single scalar for all defenders
                    if np.isscalar(vel2mag_list):
                        return float(vel2mag_list)
                    # list/array per defender
                    return float(vel2mag_list[i])

                # broadcast single velocity to all P2 players
                if (not isinstance(vel2_list, (list, tuple, np.ndarray))) or \
                   (isinstance(vel2_list, np.ndarray) and vel2_list.ndim == 1 and len(vel2_list) == 2):
                    for i, p in enumerate(pos2_list):
                        mag_i = _get_mag(i)
                        _draw_arrow(p, vel2_list, mag=mag_i, color='pink')
                else:
                    for i, (p, v) in enumerate(zip(pos2_list, vel2_list)):
                        mag_i = _get_mag(i)
                        _draw_arrow(p, v, mag=mag_i, color='pink')

    # optional time contours
    if show_time_contours:
        try:
            cs1 = ax.contour(XX, YY, TT1, levels=6,
                             colors='white', linewidths=0.6,
                             linestyles='solid', alpha=0.8)
            ax.clabel(cs1, inline=1, fontsize=8, fmt='%.1f')
            if TT_others_stack.size:
                for i in range(min(6, TT_others_stack.shape[0])):
                    cs2 = ax.contour(
                        XX, YY, TT_others_stack[i], levels=6,
                        colors='black', linewidths=0.5,
                        linestyles='dashed', alpha=0.5
                    )
                    ax.clabel(cs2, inline=1, fontsize=7, fmt='%.1f')
        except Exception:
            pass

    fig.colorbar(pcm, ax=ax)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('P1 vs P2 team control map')
    ax.set_aspect('equal')

    return fig, ax, XX, YY, TT1, TT_others_stack, combined



# ------------------------------
# convenience combine functions
# ------------------------------

def deterministic_p1_if_closest(tt1, tt_others):
    if tt_others.size == 0:
        return np.ones_like(tt1)
    min_others = np.min(tt_others, axis=0)
    return (tt1 < min_others).astype(float)

def voronoi_ownership(tt_ball, tt_def):
    if tt_def.size == 0:
        return np.ones_like(tt_ball)   # no defenders → full ownership

    min_def = np.min(tt_def, axis=0)
    ownership = (tt_ball < min_def).astype(float)
    return ownership

def soft_ownership(beta=1.0):
    def fn(tt_ball, tt_def):
        if tt_def.size == 0:
            return np.ones_like(tt_ball)   # no defenders → full ownership

        # Stack all players (ball first)
        all_times = np.concatenate([tt_ball[np.newaxis, ...], tt_def], axis=0)

        # Numerical stability: subtract per-pixel minimum before exponentiating
        exp_terms = np.exp(-beta * (all_times))

        denom = np.sum(exp_terms, axis=0) + 1e-12
        ownership = exp_terms[0] / denom   # ball-carrier channel
        return ownership
    return fn

def euclidian_time_fn(x, y):
    """Simple time-to-reach function assuming constant velocity v in straight line."""
    return np.hypot(x, y)

def transformToUpwardsMotion(x,y,vi,vt):
    #Apply a transformation matrix to make velocity vector point in the [0,1] direction
    ang=(np.pi/2 - vt)
    rot=np.array([[np.cos(ang), -np.sin(ang)],
                    [np.sin(ang), np.cos(ang)]])
    x_new, y_new = rot @ np.array([x, y])
    vi_new = vi # Assuming vi is a scalar speed, it remains unchanged by rotation
    if x_new>0: #Always CCW turn 
        x_new=-x_new
    return x_new, y_new, vi_new

def t2time(x,y,vturn,aturnmax):
    r=vturn*vturn/aturnmax
    #print(r)
    if(x**2+2*r*x+y**2<0):
        #print("Inside circle")
    #     #Descriminant negative: inside circle
    #     #need to encourage a lower vturn
        return 10*vturn+10, 10*vturn+100,0
    phi1=2*np.arctan2(-x,y+np.sqrt(x**2+2*r*x+y**2))
    phi1 = phi1 % (2 * np.pi)
    #ts=np.sqrt(x**2+2*r*x+y**2)
    #print(phi1)
    #dnew=ts*vturn
    t2= r*phi1/vturn
    newx= x - r*(1 - np.cos(phi1))
    newy= y + r*np.sin(phi1)
    dnew=np.hypot(newx,newy)
    #print(vturn,r,x,y,t2,dnew,phi1)
    return t2, dnew,phi1

def straight_line_time(d,vi,a,vmax):
    s = np.sqrt(vi*vi + 2*a*d)
    return (s - vi)/a + np.maximum(0.0, s - vmax)**2 / (2*a*vmax)

def path_time(vturn,x,y,vi,astop,ago,aturnmax,vmax):
    #print(x,y,vturn,vi)
    threshold = 0.1
    if np.abs(x)<=threshold:
        if y<0:
            #Turn and run
            t1=vi/astop
            y1= vi*t1 - 0.5*astop*t1*t1
            return vi/astop+straight_line_time(-y+y1,0,ago,vmax)
        elif y>0:
            #Go straight
            return straight_line_time(y,vi,ago,vmax)
        else:
            return 0
    if vturn>vi:
        #Accelerate
        #print("Accelerate")
        t1=(vturn-vi)/ago
        y1= vi*t1 + 0.5*ago*t1*t1
    elif vturn==vi:
        #print("Skip")
        t1=0
        y1=0
    elif vturn<vi:
        #Decelerate
        #print("Decelerate")
        t1=(vi-vturn)/astop
        y1= vi*t1 - 0.5*astop*t1*t1
    if y==y1 and x==0:
        print("HERE")
        return t1
    t2, dnew,phi1 = t2time(x, y-y1, vturn, aturnmax)
    #print(vturn,t1,t2,dnew,phi1)
    time= t1+t2 + straight_line_time(dnew,vturn,ago,vmax)
    return time
    
from scipy.optimize import minimize_scalar

def optimized_physics_time(x,y,vi,vt,astop,ago,aturnmax,vmax):
    x_new, y_new, vi_new = transformToUpwardsMotion(x, y, vi, vt)
    #print(x_new, y_new, vi_new)
    #Optimize over vturn
    res1 = minimize_scalar(lambda vturn: path_time(vturn,x_new,y_new,vi_new,astop,ago,aturnmax,vmax),
                          bounds=(0, vi), method='bounded')
    res2= minimize_scalar(lambda vturn: path_time(vturn,x_new,y_new,vi_new,astop,ago,aturnmax,vmax),
                          bounds=(vi, vmax), method='bounded')
    if res1.fun<res2.fun:
        res=res1
    else:
        res=res2
    return res.x,res.fun

def optimized_physics_time_wrapper(x,y,vi,vt,astop,ago,aturnmax,vmax):
    #print(vi)
    try:
        return optimized_physics_time(x,y,vi,vt,astop,ago,aturnmax,vmax)[1]
    except:
        print(x,y,vi,vt)
        print("Error in optimization")
        return np.nan


# fig, ax, XX, YY, TT, control = plot_control_map(lambda x,y: optimized_physics_time(x,y,vi=9,vt=0,astop=12,ago=8,aturnmax=7,vmax=10.5)[1],
#                                                xlim=(-15,15), ylim=(-15,15),
#                                                resolution=0.25, clip_positive=False, cmap='viridis',
#                                                normalize_method='none',show_contours=True)
# plt.show()



def soft_ownership(beta=1.0):
    def fn(tt_ball, tt_def):
        if tt_def.size == 0:
            return np.ones_like(tt_ball)   # no defenders → full ownership

        # Stack all players (ball first)
        all_times = np.concatenate([tt_ball[np.newaxis, ...], tt_def], axis=0)

        # Numerical stability: subtract per-pixel minimum before exponentiating
        exp_terms = np.exp(-beta * (all_times))

        denom = np.sum(exp_terms, axis=0) + 1e-12
        ownership = exp_terms[0] / denom   # ball-carrier channel
        return ownership
    return fn


def plot_control_map(time_fn,
                     xlim=(-60, 60),
                     ylim=(-30, 30),
                     resolution=0.5,
                     cmap='viridis',
                     normalize_method='inv',
                     clip_positive=True,
                     show_contours=True):
    """
    Plot a color map around (0,0) that visualizes the degree of control derived from a
    user-provided time-to-reach function.

    Args:
        time_fn: callable f(x, y) -> time (float). Interpreted as time for a player at (0,0) to reach (x,y).
        xlim, ylim: tuples giving plotting ranges (min, max).
        resolution: grid spacing in same units as x/y.
        cmap: matplotlib colormap name.
        normalize_method: how to convert time -> control. Options:
            - 'inv'   : control_raw = 1 / time
            - 'exp'   : control_raw = exp(-time)
            - 'linear': control_raw = 1 - (time - tmin) / (tmax - tmin)
            - 'none' : control_raw = time
        clip_positive: if True, negative or zero times are clipped to a small positive epsilon.
        show_contours: draw contour lines of raw time values for interpretation.

    Returns:
        fig, ax, XX, YY, TT, control  (numpy arrays & matplotlib objects)
            - TT: raw times for each grid cell
            - control: normalized control metric in [0,1] (higher => stronger control)
    """
    xs = np.arange(xlim[0], xlim[1] + 1e-9, resolution)
    ys = np.arange(ylim[0], ylim[1] + 1e-9, resolution)
    XX, YY = np.meshgrid(xs, ys)

    # Vectorize user function safely (convert returned values to float)
    vec = np.vectorize(lambda x, y: float(time_fn(x, y)))
    TT = vec(XX, YY).astype(float)

    # sanitize times
    eps = 1e-9
    if clip_positive:
        TT = np.where(TT <= 0, eps, TT)

    # convert times to a control metric in [0,1] (higher => stronger control)
    if normalize_method == 'inv':
        control_raw = 1.0 / (TT + eps)
    elif normalize_method == 'exp':
        control_raw = np.exp(-TT)
    elif normalize_method == 'linear':
        tmin = np.nanmin(TT)
        tmax = np.nanmax(TT)
        control_raw = 1.0 - (TT - tmin) / (tmax - tmin + eps)
    elif normalize_method == 'none':
        control_raw = TT
    else:
        raise ValueError('Unknown normalize_method: ' + str(normalize_method))

    # normalize to [0,1]
    cmin = np.nanmin(control_raw)
    cmax = np.nanmax(control_raw)
    control = (control_raw - cmin) / (cmax - cmin + eps)

    if normalize_method == 'none':
        control = control_raw

    # plot
    fig, ax = plt.subplots(figsize=(9, 6))
    pcm = ax.pcolormesh(XX, YY, control, cmap=cmap, shading='auto')#, vmin=0, vmax=1)
    ax.scatter(0.0, 0.0, c='k', s=60, zorder=5)
    #ax.annotate('player (0,0)', xy=(0, 0), xytext=(3, 3), fontsize=9, zorder=6)

    if show_contours:
        # contours show raw times (TT) to help interpret
        try:
            cs = ax.contour(XX, YY, TT, levels=8, colors='k', linewidths=0.6, alpha=0.6)
            ax.clabel(cs, inline=1, fontsize=8, fmt='%.2f')
        except Exception:
            pass

    fig.colorbar(pcm, ax=ax, label='normalized control (0-1)')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Control map derived from provided time-to-reach function')
    ax.set_aspect('equal')

    return fig, ax, XX, YY, TT, control


# -------------------------
# Demo helper time functions
# -------------------------

def demo_time_iso(x, y, v=7.0):
    """Isotropic: time = distance / speed (yards / yards-per-second)."""
    return np.hypot(x, y) / v


def demo_time_biased(x, y, v=7.0, prefer_dir=(1.0, 0.0), boost=2.0):
    """
    Direction-biased example: player is faster toward prefer_dir.
    prefer_dir should be a unit-like 2-tuple (dx,dy). boost controls directionality strength.
    """
    d = np.hypot(x, y)
    if d < 1e-9:
        return 0.0
    dx, dy = x / d, y / d
    proj = dx * prefer_dir[0] + dy * prefer_dir[1]
    eff_v = v * (1.0 + boost * max(0.0, proj))
    return d / eff_v

