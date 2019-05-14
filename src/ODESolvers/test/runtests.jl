using Test
using CLIMA.LowStorageRungeKuttaMethod
using CLIMA.ODESolvers

function simple_oscilator_rhs(du, u, t)
  @inbounds begin
    y, x = u[1], u[2]
    dy, dx = x, -y
    du[1], du[2] = dy, dx
  end
end
simple_oscilator_sol(t) = [cos(t);-sin(t)]

@testset "Low Storage Runge Kutta" begin
  u = simple_oscilator_sol(0)
  lsrk = LowStorageRungeKutta(simple_oscilator_rhs, u; dt = 0.00001, t0 = 0)
  timeend = 1
  solve!(u, lsrk; timeend=timeend)

  @show u - simple_oscilator_sol(timeend)
  @show u
end
