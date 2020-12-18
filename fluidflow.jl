using LinearAlgebra
using Plots
# plotly()

function add_source(N, x, s, dt)
    # for i = 1:N+2
    #     for j = 1:N+2
    #         x[i, j] += dt * s[i, j] # Add source (done manually, maybe we can make it interactive?)
    #     end
    # end
    x += dt.*s
    return x
end

function swap(a, b)
    tmp = copy(a)
    a[:, :] = b
    b[:, :] = tmp
    return a, b
end

function diffuse(N, b, x, x0, diff, dt, pb)
    # a is just a convergence factor
    a = dt * diff * N * N
    h = 1

    # Update to use dynamic error checker, instead of static # of iterations
    for k = 1:20
        for i = 2:N+1
            for j = 2:N+1
                # Gauss-seidel method, main-diagonal dominant (sum of off-diagonal terms needs to be less than diagonal)
                # x[i, j] = (x0[i, j] + a * (x[i-1, j] + x[i+1, j] + x[i, j-1] + x[i, j+1]))/(1+4*a)
                x[i,j] = (x0[i,j] + 0.50*a*(x[i+h,j] + x[i-h,j] + x[i,j+h] + x[i,j-h]) + 0.25*a*(x[i+h,j+h] + x[i-h,j+h] + x[i+h,j-h] + x[i-h,j-h]))/(1+3*a)
            end
        end
        x[:, :] = set_bnd(N, b, x, pb)
    end
    return x
end

function advect(N, b, d, d0, u, v, dt, pb)
    dt0 = dt * N
    for i = 2:N+1
        for j = 2:N+1
            # Set x and y grid coordinates
            x = i - dt0*u[i, j]
            y = j - dt0*v[i, j]
            # Enforce lower and upper bounds to constrain to grid
            if pb == 1
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
            else
                if x < 1.5
                    x = 1.5
                elseif x > N + 1.5
                    x = N + 1.5
                else
                end
                i0 = floor(Int, x) # Take the integer part
                i1 = i0 + 1
                if y < 1.5
                    y = 1.5
                elseif y > N + 1.5
                    y = N + 1.5
                else
                end
            end
            j0 = floor(Int, y)
            j1 = j0 + 1

            s1 = x - i0
            s0 = 1 - s1
            t1 = y-j0
            t0 = 1-t1

            d[i, j] = s0 * (t0*d0[i0, j0] + t1*d0[i0, j1]) + s1 * (t0*d0[i1, j0] + t1*d0[i1, j1])

        end
    end
    d[:, :] = set_bnd(N, b, d, pb)
    return d
end

function dens_step(N, x, x0, u, v, diff, dt, pb)
    x[:, :] = add_source(N, x, x0, dt)
    x0[:, :], x[:, :] = swap(x0, x) # Swap the names so that x becomes the original, and x0 becomes a guess
    x[:, :] = diffuse(N, 0, x, x0, diff, dt, pb)
    x0[:, :], x[:, :] = swap(x0, x)
    x[:, :] = advect(N, 0, x, x0, u, v, dt, pb)
    return x
end

function vel_step(N, u, u0, v, v0, visc, dt, pb)
    u[:, :] = add_source(N, u, u0, dt)
    u0[:, :], u[:, :] = swap(u0, u)
    u[:, :] = diffuse(N, 1, u, u0, visc, dt, pb)
    v[:, :] = add_source(N, v, v0, dt)
    v0[:, :], v[:, :] = swap(v0, v)
    v[:, :] = diffuse(N, 2, v, v0, visc, dt, pb)
    u[:, :], v[:, :] = project(N, u, v, zeros(xSize, ySize), zeros(xSize, ySize), pb)
    u0[:, :], u[:, :] = swap(u0, u)
    v0[:, :], v[:, :] = swap(v0, v)
    u[:, :] = advect(N, 1, u, u0, u0, v0, dt, pb)
    v[:, :] = advect(N, 2, v, v0, u0, v0, dt, pb)
    u[:, :], v[:, :] = project(N, u, v, zeros(xSize, ySize), zeros(xSize, ySize), pb)
    return u, v
end

function project(N, u, v, p, div, pb)
    h = 1.0/N
    # Calculate divergence using simple gradient
    for i=2:N+1
        for j = 2:N+1
            div[i,j] = -0.5 * h * (u[i+1, j] - u[i-1, j] + v[i, j+1] - v[i, j-1])
        end
    end
    p[:, :] .= 0

    div[:, :] = set_bnd(N, 0, div, pb)
    p[:, :] = set_bnd(N, 0, p, pb)

    # Iterate to converge to best approximation
    for k = 1:20
        for i = 2:N+1
            for j = 2:N+1
                p[i, j] = (div[i, j] + p[i-1, j] + p[i+1, j] + p[i, j-1] + p[i, j+1])/4
            end
        end
        p[:, :] = set_bnd(N, 0, p, pb)
    end

    # Subtract off converged divergence from velocity fields to obtain mass-conserving curl
    for i = 2:N+1
        for j = 2:N+1
            u[i, j] -= 0.5 * (p[i+1, j] - p[i-1, j])/h
            v[i, j] -= 0.5 * (p[i, j+1] - p[i, j-1])/h
        end
    end
    u[:, :] = set_bnd(N, 1, u, pb)
    v[:, :] = set_bnd(N, 2, v, pb)
    return u, v
end

function set_bnd(N, b, x, pb)
    if pb == 0
        for i = 2:N+1
            if b == 1 # Do left and right walls
                x[1, i] = -x[2, i]
                x[N+2, i] = -x[N+1, i]
            else
                x[1, i] = x[2, i]
                x[N+2, i] = x[N+1, i]
            end
            if b == 2 # Do top and bottom walls
                x[i, 1] = -x[i, 2]
                x[i, N+2] = -x[i, N+1]
            else
                x[i, 1] = x[i, 2]
                x[i, N+2] = x[i, N+1]
            end

            # Do the corner pieces.
            x[1, 1] = 0.5*(x[2, 1] + x[1, 2])
            x[1, N+2] = 0.5*(x[2, N+2] + x[1, N+1])
            x[N+2, 1] = 0.5*(x[N+1, 1] + x[N+2, 2])
            x[N+2, N+2] = 0.5*(x[N+1, N+2] + x[N+2, N+1])
        end
    else
        for i = 2:N+1
            if b == 1 # Do top and bottom walls
                x[1, i] = (x[N+1, i] + x[2, i])/2
                x[N+2, i] = (x[N+1, i] + x[2, i])/2
            else
                x[1, i] = x[2, i]
                x[N+2, i] = x[N+1, i]
            end
            if b == 2 # Do left and right walls
                x[i, 1] = (x[i, 2] + x[i, N+1]) / 2
                x[i, N+2] = (x[i, 2] + x[i, N+1]) / 2
            else
                x[i, 1] = x[i, 2]
                x[i, N+2] = x[i, N+1]
            end

            # Do the corner pieces.
            for i = 1:5
                x[1, 1] = 0.25*(x[2, 1] + x[1, 2] + x[1, N+2] + x[N+2, 1])
                x[1, N+2] = 0.25*(x[2, N+2] + x[1, N+1] + x[1, 1] + x[N+2, N+2])
                x[N+2, 1] = 0.25*(x[N+1, 1] + x[N+2, 2] + x[1, 1] + x[N+2, N+2])
                x[N+2, N+2] = 0.25*(x[N+1, N+2] + x[N+2, N+1] + x[1, N+2] + x[N+2, 1])
            end
        end
    end

    return x
end

##
N = 51
xSize = N + 2
ySize = N + 2

u = zeros(xSize, ySize)
v = zeros(xSize, ySize)
dens = zeros(xSize, ySize)

u_prev = zeros(xSize, ySize)
v_prev = zeros(xSize, ySize)
dens_prev = zeros(xSize, ySize)

visc = 0.01
diff = 0.01

# Enable periodic boundary conditions
pb = 0

tsteps = 200
dt = 0.01
emiss = 100
k = 200
# for tc = 1:tsteps
anim = @animate for tc = 1:tsteps
    dens_prev[26, 26] = 1000

    u_prev[:,26] .= cos(2*pi*tc/k) * emiss
    v_prev[26,:] .= cos(2*pi*tc/k) * emiss

    u[:, :], v[:, :] = vel_step(N, u, u_prev, v, v_prev, visc, dt, pb)
    dens[:, :] = dens_step(N, dens, dens_prev, u, v, diff, dt, pb)
    p1 = heatmap(1:xSize, 1:ySize, dens, clim = (0, 4), aspect_ratio=:equal)
    p2 = heatmap(1:xSize, 1:ySize, u, clim = (-3, 3), aspect_ratio=:equal)
    p3 = heatmap(1:xSize, 1:ySize, v, clim = (-3, 3), aspect_ratio=:equal)
    plot(p1, p2, p3, Layout = (1, 3))
    u_prev[:, :] = u[:, :]
    v_prev[:, :] = v[:, :]
    dens_prev[:, :] = dens[:, :]
end

gif(anim, "dens.gif", fps = 25)
