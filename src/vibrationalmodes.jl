export VibrationalMode,
    frequency,
    modestructure,
    frequency_fluctuation,
    modecutoff,
    shape,
    axis,
    frequency!,
    frequency_fluctuation!,
    modecutoff!

"""
    VibrationalMode(
            ν::Real, modestructure::Vector{Real}, δν::Union{Function,Real}=0.; N::Int=10,
            axis::NamedTuple{(:x,:y,:z)}=ẑ
        )

**user-defined fields**
* `ν::Real`: frequency [Hz]
* `modestructure::Vector{Real}`: The normalized eigenvector describing the collective motion 
        of the ions belonging to this mode.
* `δν::Union{Function,Real}`: Either a function describing time-dependent fluctuations of `ν`
        or a real number which will be converted to the constant function `t -> δν`.
* `N::Int`: Highest level included in the Hilbert space.
* `axis::NamedTuple{(:x,:y,:z)}`: The axis of symmetry for the vibration. This must lie along
        one of the basis vectors `x̂`, `ŷ` or `ẑ`.
**derived fields**
* `shape::Vector{Int}`: Indicates dimension of used Hilbert space (`=[N+1]`).
* `_cnst_δν::Bool`: A Boolean flag signifying whether or not `δν` is a constant function.

Note: the iᵗʰ Fock state (|i⟩) can be obtained by indexing as `v=VibrationalMode(...); v[i]`
"""
mutable struct VibrationalMode <: IonSimBasis
    ν::Real
    modestructure::Vector{Real}
    δν::Function
    N::Int
    shape::Vector{Int}
    axis::NamedTuple{(:x, :y, :z)}
    _cnst_δν::Bool
    function VibrationalMode(ν, modestructure; δν::Tδν=0.0, N=10, axis=ẑ) where {Tδν}
        if Tδν <: Number
            _cnst_δν = true
            δνt(t) = δν
        else
            _cnst_δν = false
            δνt = δν
        end
        return new(ν, modestructure, δνt, N, [N + 1], axis, _cnst_δν)
    end
end

#############################################################################################
# Object fields
#############################################################################################

"""
    frequency(mode::VibrationalMode)
Returns `mode.ν`
"""
frequency(mode::VibrationalMode) = mode.ν

"""
    modestructure(mode::VibrationalMode)
Returns `mode.modestructure`
"""
modestructure(mode::VibrationalMode) = mode.modestructure

"""
    frequency_fluctuation(mode::VibrationalMode)
Returns `mode.δν`
"""
frequency_fluctuation(mode::VibrationalMode) = mode.δν

"""
    modecutoff
Returns `mode.N`
"""
modecutoff(mode::VibrationalMode) = mode.N

"""
    shape(mode::VibrationalMode)
Returns `mode.shape`
"""
shape(mode::VibrationalMode) = mode.shape

"""
    axis(mode::VibrationalMode)
Returns `mode.axis`
"""
axis(mode::VibrationalMode) = mode.axis

#############################################################################################
# Setters
#############################################################################################

"""
    frequency!(mode::VibrationalMode, ν::Real)
Sets `mode.ν` to `ν`
"""
function frequency!(mode::VibrationalMode, ν::Real)
    mode.ν = ν
end

"""
    frequency_fluctuation!(mode::VibrationalMode, δν::Function)
Sets `mode.δν` to `δν`
"""
function frequency_fluctuation!(mode::VibrationalMode, δν::Function)
    mode.δν = δν
    mode._cnst_δν = false
end
"""
    frequency_fluctuation!(mode::VibrationalMode, δν::Real)
Sets `mode.δν` to a constant function `t -> δν`
"""
function frequency_fluctuation!(mode::VibrationalMode, δν::Real)
    mode.δν = (t -> δν)
    mode._cnst_δν = true
end

"""
    modecutoff!(mode::VibrationalMode, N::Int)
Sets the upper bound of the Hilbert space of `mode` to be the Fock state `N`.
"""
function modecutoff!(mode::VibrationalMode, N::Int)
    @assert N >= 0 "N must be a nonnegative integer"
    mode.N = N
    mode.shape = Int[N+1]
end


#############################################################################################
# Base functions
#############################################################################################

function Base.:(==)(b1::T, b2::T) where {T <: VibrationalMode}
    return (
        b1.ν == b2.ν &&
        b1.modestructure == b2.modestructure &&
        b1.N == b2.N &&
        b1.shape == b2.shape &&
        b1.axis == b2.axis
    )
end

# suppress long output
Base.show(io::IO, V::VibrationalMode) = print(
    io,
    "VibrationalMode(ν=$(round(V.ν,sigdigits=4)), axis=$(_print_axis(V.axis)), N=$(V.N))"
)

function Base.getindex(V::VibrationalMode, n::Int)
    @assert 0 <= n <= modecutoff(V) "n ∉ [0, $(V.N+1)]"
    return fock(hilbert_dim(V), n)
end
