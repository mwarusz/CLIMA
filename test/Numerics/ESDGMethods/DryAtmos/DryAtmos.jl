using ClimateMachine.VariableTemplates: Vars, Grad, @vars
using ClimateMachine.BalanceLaws: number_state_conservative
import ClimateMachine.BalanceLaws:
    BalanceLaw,
    vars_state_auxiliary,
    vars_state_conservative,
    vars_state_entropy,
    state_to_entropy_variables!,
    entropy_variables_to_state!,
    init_state_auxiliary!,
    init_state_conservative!,
    state_to_entropy,
    boundary_state!,
    wavespeed,
    flux_first_order!,
    source!
using StaticArrays
using LinearAlgebra: dot, I
using ClimateMachine.DGMethods.NumericalFluxes: NumericalFluxFirstOrder
import ClimateMachine.DGMethods.NumericalFluxes:
    EntropyConservative,
    numerical_volume_conservative_flux_first_order!,
    numerical_volume_fluctuation_flux_first_order!,
    ave,
    logave,
    numerical_flux_first_order!
using ClimateMachine.Orientations:
    Orientation, FlatOrientation, SphericalOrientation
using ClimateMachine.Atmos: NoReferenceState

using CLIMAParameters: AbstractEarthParameterSet
using CLIMAParameters.Planet: grav, cp_d, cv_d, planet_radius, Omega

struct EarthParameterSet <: AbstractEarthParameterSet end
const param_set = EarthParameterSet()

@inline gamma(ps::EarthParameterSet) = cp_d(ps) / cv_d(ps)

abstract type AbstractDryAtmosProblem end

struct DryAtmosModel{D, O, P, RS, S} <: BalanceLaw
    orientation::O
    problem::P
    ref_state::RS
    sources::S
end
function DryAtmosModel{D}(orientation,
                          problem::AbstractDryAtmosProblem;
                          ref_state=NoReferenceState(),
                          sources=()) where {D}
    O = typeof(orientation)
    P = typeof(problem)
    RS = typeof(ref_state)
    S = typeof(sources)
    DryAtmosModel{D, O, P, RS, S}(orientation, problem, ref_state, sources)
end

# XXX: Hack for Impenetrable.
#      This is NOT entropy stable / conservative!!!!
function boundary_state!(
    ::NumericalFluxFirstOrder,
    ::DryAtmosModel,
    state⁺,
    aux⁺,
    n,
    state⁻,
    aux⁻,
    bctype,
    _...,
)
    state⁺.ρ = state⁻.ρ
    state⁺.ρu -= 2 * dot(state⁻.ρu, n) .* SVector(n)
    state⁺.ρe = state⁻.ρe
    aux⁺.Φ = aux⁻.Φ
end

function init_state_conservative!(
    m::DryAtmosModel,
    args...,
)
  init_state_conservative!(m, m.problem, args...)
end

function init_state_auxiliary!(
    m::DryAtmosModel,
    state_auxiliary,
    geom,
)
  init_state_auxiliary!(m, m.orientation, state_auxiliary, geom)
  init_state_auxiliary!(m, m.ref_state, state_auxiliary, geom)
  init_state_auxiliary!(m, m.problem, state_auxiliary, geom)
end

function altitude(::DryAtmosModel{dim},
                  ::FlatOrientation,
                  geom) where {dim}
  @inbounds geom.coord[dim]
end

function altitude(::DryAtmosModel,
                  ::SphericalOrientation,
                  geom)
  FT = eltype(geom)
  _planet_radius::FT = planet_radius(param_set)
  norm(geom.coord) - _planet_radius
end

"""
    init_state_auxiliary!(
        m::DryAtmosModel,
        aux::Vars,
        geom::LocalGeometry
        )

Initialize geopotential for the `DryAtmosModel`.
"""
function init_state_auxiliary!(
    ::DryAtmosModel{dim},
    ::FlatOrientation,
    state_auxiliary,
    geom,
) where {dim}
    FT = eltype(state_auxiliary)
    _grav = FT(grav(param_set))
    @inbounds r = geom.coord[dim]
    state_auxiliary.Φ = _grav * r
    state_auxiliary.∇Φ = SVector{3, FT}(0, 0, _grav)
end
function init_state_auxiliary!(
    ::DryAtmosModel,
    ::SphericalOrientation,
    state_auxiliary,
    geom,
)
    FT = eltype(state_auxiliary)
    _grav = FT(grav(param_set))
    r = norm(geom.coord)
    state_auxiliary.Φ = _grav * r
    state_auxiliary.∇Φ = _grav * geom.coord / r 
end

function init_state_auxiliary!(
    ::DryAtmosModel,
    ::NoReferenceState,
    state_auxiliary,
    geom,
)
end

function init_state_auxiliary!(
    ::DryAtmosModel,
    ::AbstractDryAtmosProblem,
    state_auxiliary,
    geom,
)
end

struct DryReferenceState{TP}
  temperature_profile::TP
end
vars_state_auxiliary(::DryAtmosModel, ::DryReferenceState, FT) = @vars(T::FT, p::FT, ρ::FT, ρe::FT)
vars_state_auxiliary(::DryAtmosModel, ::NoReferenceState, FT) = @vars()

function init_state_auxiliary!(
    m::DryAtmosModel,
    ref_state::DryReferenceState,
    state_auxiliary,
    geom,
)
    FT = eltype(state_auxiliary)
    z = altitude(m, m.orientation, geom)
    T, p = ref_state.temperature_profile(param_set, z)

    _R_d::FT = R_d(param_set)
    ρ = p / (_R_d * T)
    Φ = state_auxiliary.Φ
    ρu = SVector{3, FT}(0, 0, 0)


    state_auxiliary.ref_state.T = T
    state_auxiliary.ref_state.p = p
    state_auxiliary.ref_state.ρ = ρ
    state_auxiliary.ref_state.ρe = totalenergy(ρ, ρu, p, Φ)
end

@inline function flux_first_order!(
    m::DryAtmosModel,
    flux::Grad,
    state::Vars,
    aux::Vars,
    t::Real,
    direction,
)
    ρ = state.ρ
    ρinv = 1 / ρ
    ρu = state.ρu
    ρe = state.ρe
    u = ρinv * ρu
    Φ = aux.Φ
    
    p = pressure(ρ, ρu, ρe, Φ)

    flux.ρ = ρ * u
    flux.ρu = p * I + ρ * u .* u'
    flux.ρe = u * (state.ρe + p)
end

function wavespeed(::DryAtmosModel,
                   nM,
                   state::Vars,
                   aux::Vars,
                   t::Real,
                   direction)
  ρ = state.ρ
  ρu = state.ρu
  ρe = state.ρe
  Φ = aux.Φ
  p = pressure(ρ, ρu, ρe, Φ)

  u = ρu / ρ
  uN = abs(dot(nM, u))
  return uN + soundspeed(ρ, p)
end

"""
    pressure(ρ, ρu, ρe, Φ)

Compute the pressure given density `ρ`, momentum `ρu`, total energy `ρe`, and
gravitational potential `Φ`.
"""
function pressure(ρ, ρu, ρe, Φ)
    FT = eltype(ρ)
    γ = FT(gamma(param_set))
    (γ - 1) * (ρe - dot(ρu, ρu) / 2ρ - ρ * Φ)
end

"""
    totalenergy(ρ, ρu, p, Φ)

Compute the total energy given density `ρ`, momentum `ρu`, pressure `p`, and
gravitational potential `Φ`.
"""
function totalenergy(ρ, ρu, p, Φ)
    FT = eltype(ρ)
    γ = FT(gamma(param_set))
    return p / (γ - 1) + dot(ρu, ρu) / 2ρ + ρ * Φ
end

"""
    soundspeed(ρ, p)

Compute the speed of sound from the density `ρ` and pressure `p`.
"""
function soundspeed(ρ, p)
    FT = eltype(ρ)
    γ = FT(gamma(param_set))
    sqrt(γ * p / ρ)
end

"""
    vars_state_conservative(::DryAtmosModel, FT)

The state variables for the `DryAtmosModel` are density `ρ`, momentum `ρu`,
and total energy `ρe`
"""
function vars_state_conservative(::DryAtmosModel, FT)
    @vars begin
        ρ::FT
        ρu::SVector{3, FT}
        ρe::FT
    end
end

"""
    vars_state_auxiliary(::DryAtmosModel, FT)

The auxiliary variables for the `DryAtmosModel` is gravitational potential
`Φ`
"""
function vars_state_auxiliary(m::DryAtmosModel, FT)
    @vars begin
        Φ::FT
        ∇Φ::SVector{3, FT} # TODO: only needed for the linear model
        ref_state::vars_state_auxiliary(m, m.ref_state, FT)
        problem::vars_state_auxiliary(m, m.problem, FT)
    end
end
vars_state_auxiliary(::DryAtmosModel, ::AbstractDryAtmosProblem, FT) = @vars()

"""
    vars_state_entropy(::DryAtmosModel, FT)

The entropy variables for the `DryAtmosModel` correspond to the state
variables density `ρ`, momentum `ρu`, and total energy `ρe` as well as the
auxiliary variable gravitational potential `Φ`
"""
function vars_state_entropy(::DryAtmosModel, FT)
    @vars begin
        ρ::FT
        ρu::SVector{3, FT}
        ρe::FT
        Φ::FT
    end
end

"""
    state_to_entropy_variables!(
        ::DryAtmosModel,
        entropy::Vars,
        state::Vars,
        aux::Vars,
    )

See [`BalanceLaws.state_to_entropy_variables!`](@ref)
"""
function state_to_entropy_variables!(
    ::DryAtmosModel,
    entropy::Vars,
    state::Vars,
    aux::Vars,
)
    ρ, ρu, ρe, Φ = state.ρ, state.ρu, state.ρe, aux.Φ

    FT = eltype(state)
    γ = FT(gamma(param_set))

    p = pressure(ρ, ρu, ρe, Φ)
    s = log(p / ρ^γ)
    b = ρ / 2p
    u = ρu / ρ

    entropy.ρ = (γ - s) / (γ - 1) - (dot(u, u) - 2Φ) * b
    entropy.ρu = 2b * u
    entropy.ρe = -2b
    entropy.Φ = 2ρ * b
end

"""
    entropy_variables_to_state!(
        ::DryAtmosModel,
        state::Vars,
        aux::Vars,
        entropy::Vars,
    )

See [`BalanceLaws.entropy_variables_to_state!`](@ref)
"""
function entropy_variables_to_state!(
    ::DryAtmosModel,
    state::Vars,
    aux::Vars,
    entropy::Vars,
)
    FT = eltype(state)
    β = entropy
    γ = FT(gamma(param_set))

    b = -β.ρe / 2
    ρ = β.Φ / (2b)
    ρu = ρ * β.ρu / (2b)

    p = ρ / (2b)
    s = log(p / ρ^γ)
    Φ = dot(ρu, ρu) / (2 * ρ^2) - ((γ - s) / (γ - 1) - β.ρ) / (2b)

    ρe = p / (γ - 1) + dot(ρu, ρu) / (2ρ) + ρ * Φ

    state.ρ = ρ
    state.ρu = ρu
    state.ρe = ρe
    aux.Φ = Φ
end

function state_to_entropy(::DryAtmosModel, state::Vars, aux::Vars)
    FT = eltype(state)
    ρ, ρu, ρe, Φ = state.ρ, state.ρu, state.ρe, aux.Φ
    p = pressure(ρ, ρu, ρe, Φ)
    γ = FT(gamma(param_set))
    s = log(p * ρ^γ)
    η = -ρ * s
    return η
end


function numerical_volume_conservative_flux_first_order!(
    ::EntropyConservative,
    ::DryAtmosModel,
    F::Grad,
    state_1::Vars,
    aux_1::Vars,
    state_2::Vars,
    aux_2::Vars,
)
    FT = eltype(F)
    ρ_1, ρu_1, ρe_1 = state_1.ρ, state_1.ρu, state_1.ρe
    ρ_2, ρu_2, ρe_2 = state_2.ρ, state_2.ρu, state_2.ρe
    Φ_1, Φ_2 = aux_1.Φ, aux_2.Φ
    u_1 = ρu_1 / ρ_1
    u_2 = ρu_2 / ρ_2
    p_1 = pressure(ρ_1, ρu_1, ρe_1, Φ_1)
    p_2 = pressure(ρ_2, ρu_2, ρe_2, Φ_2)
    b_1 = ρ_1 / 2p_1
    b_2 = ρ_2 / 2p_2

    ρ_avg = ave(ρ_1, ρ_2)
    u_avg = ave(u_1, u_2)
    b_avg = ave(b_1, b_2)
    Φ_avg = ave(Φ_1, Φ_2)

    usq_avg = ave(dot(u_1, u_1), dot(u_2, u_2))

    ρ_log = logave(ρ_1, ρ_2)
    b_log = logave(b_1, b_2)
    α = b_avg * ρ_log / 2b_1

    γ = FT(gamma(param_set))

    Fρ = u_avg * ρ_log
    Fρu = u_avg * Fρ' + ρ_avg / 2b_avg * I
    Fρe = (1 / (2 * (γ - 1) * b_log) - usq_avg / 2 + Φ_avg) * Fρ + Fρu * u_avg

    F.ρ += Fρ
    F.ρu += Fρu
    F.ρe += Fρe
end

function numerical_volume_fluctuation_flux_first_order!(
    ::EntropyConservative,
    ::DryAtmosModel,
    D::Grad,
    state_1::Vars,
    aux_1::Vars,
    state_2::Vars,
    aux_2::Vars,
)
    FT = eltype(D)
    ρ_1, ρu_1, ρe_1 = state_1.ρ, state_1.ρu, state_1.ρe
    ρ_2, ρu_2, ρe_2 = state_2.ρ, state_2.ρu, state_2.ρe
    Φ_1, Φ_2 = aux_1.Φ, aux_2.Φ
    p_1 = pressure(ρ_1, ρu_1, ρe_1, Φ_1)
    p_2 = pressure(ρ_2, ρu_2, ρe_2, Φ_2)
    b_1 = ρ_1 / 2p_1
    b_2 = ρ_2 / 2p_2

    ρ_log = logave(ρ_1, ρ_2)
    b_avg = ave(b_1, b_2)
    α = b_avg * ρ_log / 2b_1

    D.ρu -= α * (Φ_1 - Φ_2) * I
end

struct Coriolis end

function source!(
    m::DryAtmosModel,
    ::Coriolis,
    source,
    state_conservative,
    state_auxiliary,
)
    FT = eltype(state_conservative)
    _Omega::FT = Omega(param_set)
    # note: this assumes a SphericalOrientation
    source.ρu -= SVector(0, 0, 2 * _Omega) × state_conservative.ρu
end

function source!(
    m::DryAtmosModel,
    source,
    state_conservative,
    state_auxiliary,
)
  ntuple(Val(length(m.sources))) do s
    Base.@_inline_meta
    source!(m, m.sources[s], source, state_conservative, state_auxiliary)
  end
end


struct EntropyConservativeWithPenalty <: NumericalFluxFirstOrder end
function numerical_flux_first_order!(
    numerical_flux::EntropyConservativeWithPenalty,
    balance_law::BalanceLaw,
    fluxᵀn::Vars{S},
    normal_vector::SVector,
    state_conservative⁻::Vars{S},
    state_auxiliary⁻::Vars{A},
    state_conservative⁺::Vars{S},
    state_auxiliary⁺::Vars{A},
    t,
    direction,
) where {S, A}

    FT = eltype(fluxᵀn)
    #num_state_conservative = number_state_conservative(balance_law, FT)
    #flux = similar(fluxᵀn, Size(3, num_state_conservative))
    #numerical_volume_conservative_flux_first_order!(
    #    EntropyConservative(),
    #    balance_law,
    #    Grad{S}(flux),
    #    state_conservative⁻,
    #    state_auxiliary⁻,
    #    state_conservative⁺,
    #    state_auxiliary⁺,
    #)
    #fluxᵀn .= flux' * normal_vector

    numerical_flux_first_order!(
        EntropyConservative(),
        balance_law,
        fluxᵀn,
        normal_vector,
        state_conservative⁻,
        state_auxiliary⁻,
        state_conservative⁺,
        state_auxiliary⁺,
        t,
        direction,
    )
    fluxᵀn = parent(fluxᵀn)

    wavespeed⁻ = wavespeed(
        balance_law,
        normal_vector,
        state_conservative⁻,
        state_auxiliary⁻,
        t,
        direction,
    )
    wavespeed⁺ = wavespeed(
        balance_law,
        normal_vector,
        state_conservative⁺,
        state_auxiliary⁺,
        t,
        direction,
    )
    max_wavespeed = max.(wavespeed⁻, wavespeed⁺)
    penalty =
        max_wavespeed .*
        (parent(state_conservative⁻) - parent(state_conservative⁺))

    fluxᵀn .+= penalty / 2
end

include("linear.jl")
