using MPI
using CLIMA.Topologies
using CLIMA.Grids
using CLIMA.DGBalanceLawDiscretizations
using CLIMA.DGBalanceLawDiscretizations.NumericalFluxes
using CLIMA.MPIStateArrays
using CLIMA.LowStorageRungeKuttaMethod
using CLIMA.ODESolvers
using CLIMA.GenericCallbacks
using LinearAlgebra
using StaticArrays
using Logging, Printf, Dates

@static if Base.find_package("CuArrays") !== nothing
  using CUDAdrv
  using CUDAnative
  using CuArrays
  const ArrayTypes = VERSION >= v"1.2-pre.25" ? (CuArray,) : (Array,)
else
  const ArrayTypes = (Array, )
end

const _nstate = 5
const _ρ, _U, _V, _W, _E = 1:_nstate
const stateid = (ρid = _ρ, Uid = _U, Vid = _V, Wid = _W, Eid = _E)
const statenames = ("ρ", "U", "V", "W", "E")

using CLIMA.PlanetParameters: grav

if !@isdefined integration_testing
  const integration_testing =
    parse(Bool, lowercase(get(ENV,"JULIA_CLIMA_INTEGRATION_TESTING","false")))
  using Random
end

using CLIMA.PlanetParameters: Omega, grav, planet_radius

include("mms_solution_generated.jl")

# preflux computation
@inline function preflux(Q, _...)
  γ::eltype(Q) = γ_exact
  @inbounds ρ, U, V, W, E = Q[_ρ], Q[_U], Q[_V], Q[_W], Q[_E]
  ρinv = 1 / ρ
  u, v, w = ρinv * U, ρinv * V, ρinv * W
  ((γ-1)*(E - ρinv * (U^2 + V^2 + W^2) / 2), u, v, w, ρinv)
end

eulerflux!(F, Q, QV, aux, t) =
eulerflux!(F, Q, QV, aux, t, preflux(Q)...)

@inline function eulerflux!(F, Q, QV, aux, t, P, u, v, w, ρinv)
  @inbounds begin
    ρ, U, V, W, E = Q[_ρ], Q[_U], Q[_V], Q[_W], Q[_E]

    F[1, _ρ], F[2, _ρ], F[3, _ρ] = U          , V          , W
    F[1, _U], F[2, _U], F[3, _U] = u * U  + P , v * U      , w * U
    F[1, _V], F[2, _V], F[3, _V] = u * V      , v * V + P  , w * V
    F[1, _W], F[2, _W], F[3, _W] = u * W      , v * W      , w * W + P
    F[1, _E], F[2, _E], F[3, _E] = u * (E + P), v * (E + P), w * (E + P)
  end
end

# max eigenvalue
@inline function wavespeed(n, Q, aux, t, P, u, v, w, ρinv)
  γ::eltype(Q) = γ_exact
  @inbounds abs(n[1] * u + n[2] * v + n[3] * w) + sqrt(ρinv * γ * P)
end

const _nauxstate = 7
const _a_ϕ, _a_ϕx, _a_ϕy, _a_ϕz, _a_x, _a_y, _a_z = 1:_nauxstate
@inline function auxiliary_state_initialization!(aux, x, y, z)
  @inbounds begin
    aux[_a_ϕ]  = grav * hypot(x, y, z)
    aux[_a_ϕx] = 0
    aux[_a_ϕy] = 0
    aux[_a_ϕz] = 0
    aux[_a_x] = x
    aux[_a_y] = y
    aux[_a_z] = z
  end
end

@inline function source!(S, Q, aux, t)
  @inbounds begin
    # ϕx, ϕy, ϕz = aux[_ϕx], aux[_ϕy], aux[_ϕz]
    # ρ = Q[_ρ]
    # S[_ρ] = 0
    # S[_U] = 0 #-ρ * ϕx
    # S[_V] = 0 #-ρ * ϕy
    # S[_W] = 0 #-ρ * ϕz
    # S[_E] = 0
    x,y,z = aux[_a_x], aux[_a_y], aux[_a_z]
    r = hypot(x, y, z)
    cπt = cos(π * t)
    sπt = sin(π * t)
    S[_ρ] = Sρ_g(t, x, y, z, r, cπt, sπt)
    S[_U] = SU_g(t, x, y, z, r, cπt, sπt)
    S[_V] = SV_g(t, x, y, z, r, cπt, sπt)
    S[_W] = SW_g(t, x, y, z, r, cπt, sπt)
    S[_E] = SE_g(t, x, y, z, r, cπt, sπt)
  end
end

function initialcondition!(Q, t, x, y, z, _...)
  DFloat = eltype(Q)
  r = hypot(x, y, z)
  cπt = cos(π * t)
  sπt = sin(π * t)
  ρ::DFloat = ρ_g(t, x, y, z, r, cπt, sπt)
  U::DFloat = U_g(t, x, y, z, r, cπt, sπt)
  V::DFloat = V_g(t, x, y, z, r, cπt, sπt)
  W::DFloat = W_g(t, x, y, z, r, cπt, sπt)
  E::DFloat = E_g(t, x, y, z, r, cπt, sπt)

  if integration_testing
    @inbounds Q[_ρ], Q[_U], Q[_V], Q[_W], Q[_E] = ρ, U, V, W, E
  else
    @inbounds Q[_ρ], Q[_U], Q[_V], Q[_W], Q[_E] =
    10+rand(), rand(), rand(), rand(), 10+rand()
  end
end

@inline function bcstate!(QP, QVP, auxP, nM, QM, QVM, auxM, bctype, t, _...)
  @inbounds begin
    x, y, z = auxM[_a_x], auxM[_a_y], auxM[_a_z]
    if integration_testing
      initialcondition!(QP, t, x, y, z)
    else
      for s = 1:length(QP)
        QP[s] = QM[length(QP)+1-s]
      end
      for s = 1:_nviscstates
        QVP[s] = QVM[s]
      end
    end
  end
  nothing
end

function run(mpicomm, ArrayType, topl, warpfun, N, timeend, DFloat, dt)

  grid = DiscontinuousSpectralElementGrid(topl,
                                          FloatType = DFloat,
                                          DeviceArray = ArrayType,
                                          polynomialorder = N,
                                          meshwarp = warpfun,
                                         )

  # spacedisc = data needed for evaluating the right-hand side function
  numflux!(x...) = NumericalFluxes.rusanov!(x..., eulerflux!, wavespeed,
                                            preflux)
  numbcflux!(x...) = NumericalFluxes.rusanov_boundary_flux!(x..., eulerflux!,
                                                            bcstate!,
                                                            wavespeed, preflux)
  spacedisc = DGBalanceLaw(grid = grid,
                           length_state_vector = _nstate,
                           flux! = eulerflux!,
                           numerical_flux! = numflux!,
                           numerical_boundary_flux! = numbcflux!,
                           auxiliary_state_length = _nauxstate,
                           auxiliary_state_initialization! =
                           auxiliary_state_initialization!,
                           source! = source!)
  DGBalanceLawDiscretizations.grad_auxiliary_state!(spacedisc, _a_ϕ,
                                                    (_a_ϕx, _a_ϕy, _a_ϕz))

  # This is a actual state/function that lives on the grid
  initialcondition(Q, x...) = initialcondition!(Q, DFloat(0), x...)
  Q = MPIStateArray(spacedisc, initialcondition)
  DGBalanceLawDiscretizations.writevtk("initcond", Q, spacedisc, statenames)

  lsrk = LowStorageRungeKutta(spacedisc, Q; dt = dt, t0 = 0)

  eng0 = norm(Q)
  @info @sprintf """Starting
  norm(Q₀) = %.16e""" eng0

  # Set up the information callback
  starttime = Ref(now())
  cbinfo = GenericCallbacks.EveryXWallTimeSeconds(60, mpicomm) do (s=false)
    if s
      starttime[] = now()
    else
      energy = norm(Q)
      @info @sprintf("""Update
                     simtime = %.16e
                     runtime = %s
                     norm(Q) = %.16e""", ODESolvers.gettime(lsrk),
                     Dates.format(convert(Dates.DateTime,
                                          Dates.now()-starttime[]),
                                  Dates.dateformat"HH:MM:SS"),
                     energy)
    end
  end

  step = [0]
  mkpath("vtk")
  cbvtk = GenericCallbacks.EveryXSimulationSteps(100) do (init=false)
    outprefix = @sprintf("vtk/cns_sphere_mpirank%04d_step%04d",
                         MPI.Comm_rank(mpicomm), step[1])
    @debug "doing VTK output" outprefix
    DGBalanceLawDiscretizations.writevtk(outprefix, Q, spacedisc, statenames)
    step[1] += 1
    nothing
  end

  # solve!(Q, lsrk; timeend=timeend, callbacks=(cbinfo, ))
  solve!(Q, lsrk; timeend=timeend, callbacks=(cbinfo, cbvtk))


  # Print some end of the simulation information
  engf = norm(Q)
  if integration_testing
    Qe = MPIStateArray(spacedisc,
                       (Q, x...) -> initialcondition!(Q, DFloat(timeend), x...))
    engfe = norm(Qe)
    errf = euclidean_distance(Q, Qe)
    @info @sprintf """Finished
    norm(Q)                 = %.16e
    norm(Q) / norm(Q₀)      = %.16e
    norm(Q) - norm(Q₀)      = %.16e
    norm(Q - Qe)            = %.16e
    norm(Q - Qe) / norm(Qe) = %.16e
    """ engf engf/eng0 engf-eng0 errf errf / engfe
  else
    error()
  end
  integration_testing ? errf : (engf / eng0)
end

using Test
let
  MPI.Initialized() || MPI.Init()
  Sys.iswindows() || (isinteractive() && MPI.finalize_atexit())
  mpicomm = MPI.COMM_WORLD
  ll = uppercase(get(ENV, "JULIA_LOG_LEVEL", "INFO"))
  loglevel = ll == "DEBUG" ? Logging.Debug :
  ll == "WARN"  ? Logging.Warn  :
  ll == "ERROR" ? Logging.Error : Logging.Info
  logger_stream = MPI.Comm_rank(mpicomm) == 0 ? stderr : devnull
  global_logger(ConsoleLogger(logger_stream, loglevel))
  @static if Base.find_package("CUDAnative") !== nothing
    device!(MPI.Comm_rank(mpicomm) % length(devices()))
  end

  polynomialorder = 4
  base_Nhorz = 4
  base_Nvert = 2
  Rinner = 1//2
  Router = 1
  if integration_testing
    lvls = 3
  else
    error()
  end


  for ArrayType in ArrayTypes
    for DFloat in (Float64,) #Float32)
      result = zeros(DFloat, lvls)
      for l = 1:lvls
        integration_testing || Random.seed!(0)
        Nhorz = 2^(l-1) * base_Nhorz
        Nvert = 2^(l-1) * base_Nvert
        Rrange = range(DFloat(Rinner); length=Nvert+1, stop=Router)
        topl = StackedCubedSphereTopology(mpicomm, Nhorz, Rrange)
        dt = 5e-4 / Nhorz
        warpfun = Topologies.cubedshellwarp
        timeend = integration_testing ? 0.01 : 2dt
        nsteps = ceil(Int64, timeend / dt)
        dt = timeend / nsteps

        @info (ArrayType, DFloat, "sphere")
        result[l] = run(mpicomm, ArrayType, topl, warpfun,
                        polynomialorder, timeend, DFloat, dt)
        if integration_testing
          # @test result[l] ≈ DFloat(expected_result[l])
        else
          @test result[l] ≈ expected_result[MPI.Comm_size(mpicomm), DFloat]
        end
      end
      if integration_testing
        @info begin
          msg = ""
          for l = 1:lvls-1
            rate = log2(result[l]) - log2(result[l+1])
            msg *= @sprintf("\n  rate for level %d = %e\n", l, rate)
          end
          msg
        end
      end
    end
  end
end

isinteractive() || MPI.Finalize()

nothing
