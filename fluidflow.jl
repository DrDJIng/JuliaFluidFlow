using LinearAlgebra
using Plots
# using PyPlot
# plotly()

function add_source(N, x, s, dt)
    x += dt.*s
    return x
end

function swap(a, b)
    tmp = copy(a)
    a[:, :, :] = b
    b[:, :, :] = tmp
    return a, b
end

function diffuse(N, b, x, x0, diff, dt, pb)
    # a is just a convergence factor
    a = dt * diff * N * N * N
    h = 1

    # Update to use dynamic error checker, instead of static # of iterations
    for k = 1:20
        for i = 2:N+1
            for j = 2:N+1
                for k = 2:N+1
                    # Gauss-seidel method, main-diagonal dominant (sum of off-diagonal terms needs to be less than diagonal)
                    x[i, j, k] = (x0[i, j, k] + a * (x[i-1, j, k] + x[i+1, j, k] + x[i, j-1, k] + x[i, j+1, k] + x[i, j, k+1] + x[i, j, k-1]))/(1+6*a)
                    # x[i,j] = (x0[i,j] + 0.50*a*(x[i+h,j] + x[i-h,j] + x[i,j+h] + x[i,j-h]) + 0.25*a*(x[i+h,j+h] + x[i-h,j+h] + x[i+h,j-h] + x[i-h,j-h]))/(1+3*a)
                end
            end
        end
        x[:, :, :] = set_bnd(N, b, x, pb)
    end
    return x
end

function advect(N, b, d, d0, u, v, w, dt, pb)
    dt0 = dt * N
    for i = 2:N+1
        for j = 2:N+1
            for k = 2:N+1
                # Set x and y grid coordinates
                x = i - dt0*u[i, j, k]
                y = j - dt0*v[i, j, k]
                z = k - dt0*w[i, j, k]
                # Enforce lower and upper bounds to constrain to grid
                if x < 1.5
                    x = N + 1.5
                elseif x > N + 1.5
                    x = 1.5
                else
                end
                i0 = floor(Int, x) # Take the integer part
                i1 = i0 + 1
                if y < 1.5
                    y = N + 1.5
                elseif y > N + 1.5
                    y = 1.5
                else
                end
                j0 = floor(Int, y)
                j1 = j0 + 1
                if z < 1.5
                    z = N + 1.5
                elseif z > N + 1.5
                    z = 1.5
                else
                end
                k0 = floor(Int, y)
                k1 = k0 + 1


                s1 = x - i0
                s0 = 1 - s1
                t1 = y - j0
                t0 = 1 - t1
                u1 = z - k0
                u0 = 1 - u1

                d[i, j, k] = u0*(s0 * (t0*d0[i0, j0, k0] + t1*d0[i0, j1, k0]) + s1 * (t0*d0[i1, j0, k0] + t1*d0[i1, j1, k0]))
                            + u1*(s0 * (t0*d0[i0, j0, k1] + t1*d0[i0, j1, k1]) + s1 * (t0*d0[i1, j0, k1] + t1*d0[i1, j1, k1]))
            end
        end
    end
    d[:, :, :] = set_bnd(N, b, d, pb)
    return d
end

function dens_step(N, x, x0, u, v, w, diff, dt, pb)
    x[:, :, :] = add_source(N, x, x0, dt)
    x0[:, :, :], x[:, :, :] = swap(x0, x) # Swap the names so that x becomes the original, and x0 becomes a guess
    x[:, :, :] = diffuse(N, 0, x, x0, diff, dt, pb)
    x0[:, :, :], x[:, :, :] = swap(x0, x)
    x[:, :, :] = advect(N, 0, x, x0, u, v, w, dt, pb)
    return x
end

function vel_step(N, u, u0, v, v0, w, w0, visc, dt, pb)
    u[:, :, :] = add_source(N, u, u0, dt)
    u0[:, :, :], u[:, :, :] = swap(u0, u)
    u[:, :, :] = diffuse(N, 1, u, u0, visc, dt, pb)
    v[:, :, :] = add_source(N, v, v0, dt)
    v0[:, :, :], v[:, :, :] = swap(v0, v)
    v[:, :, :] = diffuse(N, 2, v, v0, visc, dt, pb)
    w[:, :, :] = add_source(N, w, w0, dt)
    w0[:, :, :], w[:, :, :] = swap(w0, w)
    w[:, :, :] = diffuse(N, 3, w, w0, visc, dt, pb)
    u[:, :, :], v[:, :, :], w[:, :, :] = project(N, u, v, w, zeros(xSize, ySize, zSize), zeros(xSize, ySize, zSize), pb)
    u0[:, :, :], u[:, :, :] = swap(u0, u)
    v0[:, :, :], v[:, :, :] = swap(v0, v)
    w0[:, :, :], w[:, :, :] = swap(w0, w)
    u[:, :, :] = advect(N, 1, u, u0, u0, v0, w0, dt, pb)
    v[:, :, :] = advect(N, 2, v, v0, u0, v0, w0, dt, pb)
    w[:, :, :] = advect(N, 3, w, w0, u0, v0, w0, dt, pb)
    u[:, :, :], v[:, :, :], w[:, :, :] = project(N, u, v, w,zeros(xSize, ySize, zSize), zeros(xSize, ySize, zSize), pb)
    return u, v, w
end

function project(N, u, v, w, p, div, pb)
    h = 1.0/N
    # Calculate divergence using simple gradient
    for i=2:N+1
        for j = 2:N+1
            for k = 2:N+1
                div[i, j, k] = -0.5 * h * (u[i+1, j, k] - u[i-1, j, k] + v[i, j+1, k] - v[i, j-1, k] + w[i, j, k+1] - w[i, j, k-1])
            end
        end
    end
    p[:, :, :] .= 0

    div[:, :, :] = set_bnd(N, 0, div, pb)
    p[:, :, :] = set_bnd(N, 0, p, pb)

    # Iterate to converge to best approximation
    for k = 1:40
        for i = 2:N+1
            for j = 2:N+1
                for k = 2:N+1
                    # p[i, j] = (div[i, j] + p[i-1, j] + p[i+1, j] + p[i, j-1] + p[i, j+1])/4
                    p[i, j, k] = (div[i, j, k] + p[i-1, j, k] + p[i+1, j, k] + p[i, j-1, k] + p[i, j+1, k] + p[i, j, k+1] + p[i, j, k-1])/6
                end
            end
        end
        p[:, :, :] = set_bnd(N, 0, p, pb)
    end

    # Subtract off converged divergence from velocity fields to obtain mass-conserving curl
    for i = 2:N+1
        for j = 2:N+1
            for k = 2:N+1
                u[i, j, k] -= 0.5 * (p[i+1, j, k] - p[i-1, j, k])/h
                v[i, j, k] -= 0.5 * (p[i, j+1, k] - p[i, j-1, k])/h
                w[i, j, k] -= 0.5 * (p[i, j, k+1] - p[i, j, k-1])/h
            end
        end
    end
    u[:, :, :] = set_bnd(N, 1, u, pb)
    v[:, :, :] = set_bnd(N, 2, v, pb)
    w[:, :, :] = set_bnd(N, 3, w, pb)
    return u, v, w
end

function set_bnd(N, b, x, pb)
    for i = 2:N+1
        if b == 1 # Do left and right walls
            x[1, i, i] = -x[2, i, i]
            x[N+2, i, i] = -x[N+1, i, i]
        else
            x[1, i, i] = x[2, i, i]
            x[N+2, i, i] = x[N+1, i, i]
        end
        if b == 2 # Do top and bottom walls
            x[i, 1, i] = -x[i, 2, i]
            x[i, N+2, i] = -x[i, N+1, i]
        else
            x[i, 1, i] = x[i, 2, i]
            x[i, N+2, i] = x[i, N+1, i]
        end
        if b == 3 # Do Z walls
            x[i, i, 1] = -x[i, 2, i]
            x[i, N+2, i] = -x[i, N+1, i]
        else
            x[i, i, 1] = x[i, i, 1]
            x[i, i, N+2] = x[i, i, N+1]
        end


        # Do the corner pieces.
        # x[1, 1, 1] = 0.5*(x[2, 1, 1] + x[1, 2, 1] + x[1, 1, 2])
        # x[N+2, 1, 1] = 0.5*(x[N+1, 1, 1] + x[N+2, 2, 1] + x[N+2, 1, 2])
        # x[1, N+2, 1] = 0.5*(x[2, N+2] + x[1, N+1])
        # x[1, 1, N+2] = 0.5*(x[2, N+2] + x[1, N+1])
        # x[N+2, N+2, 1] = 0.5*(x[N+1, 1] + x[N+2, 2])
        # x[1, N+2, N+2] = 0.5*(x[N+1, 1] + x[N+2, 2])
        # x[N+2, 1, N+2] = 0.5*(x[N+1, 1] + x[N+2, 2])
        # x[N+2, N+2, N+2] = 0.5*(x[N+1, N+2] + x[N+2, N+1])
    end
    return x
end

##
N = 51
xSize = N + 2
ySize = N + 2
zSize = N + 2

u = zeros(xSize, ySize, zSize)
v = zeros(xSize, ySize, zSize)
w = zeros(xSize, ySize, zSize)
dens = zeros(xSize, ySize, zSize)

u_prev = zeros(xSize, ySize, zSize)
v_prev = zeros(xSize, ySize, zSize)
w_prev = zeros(xSize, ySize, zSize)
dens_prev = zeros(xSize, ySize, zSize)
dens_prev[:, :, 26] .= 1
w_prev[:, :, :] .= -0.1

visc = 1
diff = 0.01

# Enable periodic boundary conditions
pb = 0

tsteps = 30
dt = 0.001
k = 200

# for tc = 1:tsteps
anim = @animate for tc = 1:tsteps
    global dens_prev, u_prev, v_prev
    # cos(2*pi*tc/k) *

    u[:, :, :], v[:, :, :] = vel_step(N, u, u_prev, v, v_prev, w, w_prev, visc, dt, pb)
    dens[:, :, :] = dens_step(N, dens, dens_prev, u, v, w, diff, dt, pb)
    p1 = heatmap(1:xSize, 1:ySize, dens[:,:,26], clim = (0, 1), aspect_ratio=:equal, title = "Density")
    p2 = heatmap(1:xSize, 1:ySize, u[:,:,26], clim = (-3, 3), aspect_ratio=:equal, title = "U-velocity")
    p3 = heatmap(1:xSize, 1:ySize, v[:,:,26], clim = (-3, 3), aspect_ratio=:equal, title = "V-velocity")
    plot(p1, p2, p3, Layout = (1, 3))
    u_prev[:, :, :] = u[:, :, :]
    v_prev[:, :, :] = v[:, :, :]
    dens_prev[:, :, :] = dens[:, :, :]
end

gif(anim, "dens.gif", fps = 5)
##
# anim = @animate for zc = 1:zSize
#     heatmap(1:xSize, 1:ySize, dens[:, :, zc], clim = (-100, 100))
# end
# gif(anim, "dens.gif", fps = 25)
