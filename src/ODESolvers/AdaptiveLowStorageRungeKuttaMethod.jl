module AdaptiveLowStorageRungeKuttaMethod
export AdaptiveLowStorageRungeKutta, updatedt!

using Requires

@init @require CuArrays = "3a865a2d-5b23-5a0f-bc46-62713ec82fae" begin
  using .CuArrays
  using .CuArrays.CUDAnative
  using .CuArrays.CUDAnative.CUDAdrv

  include("LowStorageRungeKuttaMethod_cuda.jl")
end

using ..ODESolvers
ODEs = ODESolvers
using ..SpaceMethods

"""
    AdaptiveLowStorageRungeKutta(f, Q; t0 = 0)

This is a time stepping object for explicitly time stepping the differential
equation given using a 3S* embedded Low Storage RK scheme [Ketcheson2010JCP].

```
S3 := un
S1 := un
for i = 2:m+1 do
  S2 := S2 + d * S1
  S1 := gi1 S1 + gi2 * S2 + gi3 S3 + b dt F(S1)
end
S2 := (S2 + dm1 S1 + dm2 S3) / sum(delta)
un1 := S1
```

### References

 @article{Ketcheson2010JCP,
   title={{Runge-Kutta} methods with minimum storage implementations},
   author={David I Ketcheson},
   journal={Journal of Computational Physics},
   volume={229},
   number={5},
   pages={1763--1773},
   year={2010},
   doi={10.1016/j.jcp.2009.11.006}
 }
"""
struct AdaptiveLowStorageRungeKutta{T, AT, Nstages} <: ODEs.AbstractODESolver
  "time step"
  dt::Array{T,1}
  "time"
  t::Array{T,1}
  "rhs function"
  rhs!::Function
  "Storage for RHS during the AdaptiveLowStorageRungeKutta update"
  dQ::AT
  "Storage for error estimator"
  Q2::AT
  "Storage for previous step"
  Q3::AT
  "low storage RK coefficient vector γ1 coefficients (rhs scaling)"
  ECRKγ1::NTuple{Nstages, T}
  "low storage RK coefficient vector γ2 coefficients (rhs scaling)"
  ECRKγ2::NTuple{Nstages, T}
  "low storage RK coefficient vector γ2 coefficients (rhs scaling)"
  ECRKγ3::NTuple{Nstages, T}
  "low storage RK coefficient vector β (rhs add in scaling for S1)"
  ECRKβ::NTuple{Nstages,  T}
  "low storage RK coefficient vector δ (rhs combination scaling S2 = S2 + δ S1)"
  ECRKδ::NTuple{Nstages+2, T}
  "low storage RK coefficient vector C (time scaling)"
  ECRKC::NTuple{Nstages, T}
  "sum(ECRKδ)"
  sum_ECRKδ::T
  function AdaptiveLowStorageRungeKutta(rhs!::Function, Q::AT; dt=nothing,
                                        t0=0) where {AT<:AbstractArray}

    @assert dt != nothing

    T = eltype(Q)
    dt = [T(dt)]
    t0 = [T(t0)]

    # Coefficients from Ketcheson2010JCP, Coefficients for RK4(3)5[3S*]
    ECRKγ1 = (
              T(+0.000000000000000),
              T(-0.497531095840104),
              T(+1.010070514199942),
              T(-3.196559004608766),
              T(+1.717835630267259),
             )
    ECRKγ2 = (
              T(+1.000000000000000),
              T(+1.384996869124138),
              T(+3.878155713328178),
              T(-2.324512951813145),
              T(-0.514633322274467),
             )
    ECRKγ3 = (
              T(+0.000000000000000),
              T(+0.000000000000000),
              T(+0.000000000000000),
              T(+1.642598936063715),
              T(+0.188295940828347),
             )
    ECRKβ = (
             T(+0.075152045700771),
             T(+0.211361016946069),
             T(+1.100713347634329),
             T(+0.728537814675568),
             T(+0.393172889823198),
            )
    ECRKδ = (
             T(+1.000000000000000),
             T(+0.081252332929194),
             T(-1.083849060586449),
             T(-1.096110881845602),
             T(+2.859440022030827),
             T(-0.655568367959557),
             T(-0.194421504490852),
            )
    ECRKC = (
             T(+0.000000000000000),
             T(+0.075152045700771),
             T(+0.182427714642998),
             T(+0.541854428334307),
             T(+0.822490156566877),
            )
    sum_ECRKδ = sum(ECRKδ)

    new{T, AT, length(RKA)}(dt, t0, rhs!, similar(Q), RKA, RKB, RKC)
  end
end

function AdaptiveLowStorageRungeKutta(spacedisc::AbstractSpaceMethod, Q;
                                      dt=nothing, t0=0)
  AdaptiveLowStorageRungeKutta((x...) -> SpaceMethods.odefun!(spacedisc, x...),
                               Q; dt=dt, t0=t0)
end


function ODEs.dostep!(Q1, eclsrk::AdaptiveLowStorageRungeKutta, timeend,
                      adjustfinalstep)
  time, dt = eclsrk.t[1], eclsrk.dt[1]
  if adjustfinalstep && time + dt > timeend
    dt = timeend - time
    @assert dt > 0
  end
  ECRKγ1, ECRKγ2, ECRKγ3 = eclsrk.ECRKγ1, eclsrk.ECRKγ2, eclsrk.ECRKγ3
  ECRKβ, ECRKδ, ECRKC = eclsrk.ECRKβ, eclsrk.ECRKδ, eclsrk.ECRKC
  rhs!, dQ = eclsrk.rhs!, eclsrk.dQ
  Q2, Q3 = eclsrk.Q2, eclsrk.Q3

  # init stage
  Q2.Q .= 0
  Q3.Q .= Q1.Q

  for s = 1:length(ECRKγ1)
    rhs!(dQ, Q1, time)

    # update solution and scale RHS
    # FIXME: GPUify
    update!(Val(size(Q1,2)), Val(size(Q1,1)), dQ.Q,
            Q1.Q, Q2.Q, Q3.Q, Q1.realelems,
            ECRKγ1[s], ECRKγ2[s], ECRKγ3[s],
            dt * ECRKβ[s], ECRKδ[s])

    time += ECRKC[s] * dt
  end

  c1 = ECRKδ[end-1] / eclsrk.sum_ECRKδ - 1
  c2 = 1 / eclsrk.sum_ECRKδ
  c3 = ECRKδ[end] / eclsrk.sum_ECRKδ
  ΔQ = Q2
  error_estimate!(Val(size(Q1,2)), Val(size(Q1,1)), ΔQ.Q Q1.Q, Q2.Q, Q3.Q,
                  Q1.realelems, c1, c2, c3)

  if dt == eclsrk.dt[1]
    eclsrk.t[1] += dt
  else
    eclsrk.t[1] = timeend
  end

end

# {{{ Update solution (for all dimensions)
function initstage!(::Val{nstates}, ::Val{Np}, Q1, Q2, Q3) where {nstates, Np}
  @inbounds for e = elems, s = 1:nstates, i = 1:Np
    Q3[i, s, e] = Q1[i, s, e]
  end
end
# }}}

# {{{ Update solution (for all dimensions)
function update!(::Val{nstates}, ::Val{Np}, rhs, Q1, Q2, Q3, elems,
                 γ1, γ2, γ3, dtβ, δ) where {nstates, Np, T}
  @inbounds for e = elems, s = 1:nstates, i = 1:Np
    Q2[i, s, e] += δ * Q1[i, s, e]
    Q1[i, s, e] = γ1 * Q1[i, s, e] + γ2 * Q2[i, s, e] + γ3 * Q3[i, s, e] +
                  dtβ * rhs[i, s, e]
  end
end
# }}}

# {{{ Update solution (for all dimensions)
function error_estimate!(::Val{nstates}, ::Val{Np}, ΔQ, Q1, Q2, Q3, elems,
                         c1, c2, c3) where {nstates, Np, T}
  @inbounds for e = elems, s = 1:nstates, i = 1:Np
    ΔQ[i, s, e] = c1 * Q1[i, s, e] + c2 * Q2[i, s, e] +  + c3 * Q3[i, s, e]
  end
end
# }}}

end
