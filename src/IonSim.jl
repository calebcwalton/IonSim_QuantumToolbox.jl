module IonSim

using QuantumToolbox
using SparseArrays
import OrderedCollections: OrderedDict

export OrderedDict, analytical
# Export commonly used QuantumToolbox.jl functions and IonSim aliases
export ⊗,
    tensor,
    normalize,
    normalize!,
    expect,
    tr,
    ptrace,
    entropy_vn,
    fidelity,
    exp,
    norm,
    dag,
    dagger,
    dm,
    ket2dm

# Aliases for backward compatibility with QuantumOptics.jl naming
const dagger = dag
const dm = ket2dm

# used for copying composite types
Base.copy(x::T) where {T} = T([getfield(x, k) for k in fieldnames(T)]...)

"""
    IonSimBasis
An abstract type for specialized bases, which are unique to IonSim.
"""
abstract type IonSimBasis end
export IonSimBasis

"""
    hilbert_dim(b::IonSimBasis)
Returns the integer dimension of this basis element's Hilbert space.
"""
hilbert_dim(b::IonSimBasis) = b.shape[1]
export hilbert_dim

"""
    IonSimCompositeBasis
A composite basis for IonSim, storing the component bases and their dimensions.
"""
struct IonSimCompositeBasis
    components::Vector{IonSimBasis}
    dims::Vector{Int}
end
export IonSimCompositeBasis

hilbert_dim(b::IonSimCompositeBasis) = prod(b.dims)

# Allow ⊗ on IonSimBasis objects to produce IonSimCompositeBasis
Base.kron(a::IonSimBasis, b::IonSimBasis) = IonSimCompositeBasis(
    IonSimBasis[a, b], [hilbert_dim(a), hilbert_dim(b)]
)
Base.kron(a::IonSimCompositeBasis, b::IonSimBasis) = IonSimCompositeBasis(
    IonSimBasis[a.components..., b], [a.dims..., hilbert_dim(b)]
)
Base.kron(a::IonSimBasis, b::IonSimCompositeBasis) = IonSimCompositeBasis(
    IonSimBasis[a, b.components...], [hilbert_dim(a), b.dims...]
)

"""
    _zero_op(b::IonSimCompositeBasis)
Create a zero sparse QuantumObject with composite dimensions matching `b`.
Uses reversed kron ordering to match the Hamiltonian index convention.
"""
function _zero_op(b::IonSimCompositeBasis)
    op = sparse(tensor(reverse([qeye(d) for d in b.dims])...))
    op.data .= 0
    return op
end

"""
    _embed(b::IonSimCompositeBasis, positions::Vector{Int}, ops)
Embed local operators into the full composite Hilbert space.
Uses reversed kron ordering to match the Hamiltonian index convention.
"""
function _embed(b::IonSimCompositeBasis, positions::Vector{Int}, ops)
    factors = [qeye(d) for d in b.dims]
    for (pos, op) in zip(positions, ops)
        factors[pos] = op
    end
    return tensor(reverse(factors)...)
end

"""
    _ionsim_tensor(args...)
Tensor product using the internal IonSim convention (reversed kron order,
matching QuantumOptics.jl's convention). Used for building composite states
and operators that are consistent with the Hamiltonian.
"""
_ionsim_tensor(args...) = tensor(reverse(args)...)
_ionsim_tensor(args::Vector) = tensor(reverse(args)...)

"""
    iontensor(args...)
Tensor product for IonSim composite states and operators.
Use this instead of `⊗`/`tensor` when building initial states for time evolution.
"""
iontensor(args...) = _ionsim_tensor(args...)
export iontensor

include("constants.jl")
include("ions.jl")
include("vibrationalmodes.jl")
include("lasers.jl")
include("iontraps.jl")
include("chambers.jl")
include("operators.jl")
include("hamiltonians.jl")
include("timeevolution.jl")
include("species/_include_species.jl")

module analytical
include("analyticfunctions.jl")
end

end  # main module
