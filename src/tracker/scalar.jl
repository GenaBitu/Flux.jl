struct TrackedReal{T<:Real} <: Real
  data::T
  tracker::Tracked{T}
end

TrackedReal(x::Real) = TrackedReal(x, Tracked{typeof(x)}(Call(), zero(x)))

data(x::TrackedReal) = x.data
tracker(x::TrackedReal) = x.tracker

track(f::Call, x::Real) = TrackedReal(x, Tracked{typeof(x)}(f, zero(x)))

function back!(x::TrackedReal)
    isinf(x) && error("Loss is Inf")
    isnan(x) && error("Loss is NaN")
    return back!(x, 1)
end

function Base.show(io::IO, x::TrackedReal)
  show(io, data(x))
  print(io, " (tracked)")
end

Base.decompose(x::TrackedReal) = Base.decompose(data(x))

Base.convert(::Type{TrackedReal{T}}, x::TrackedReal{T}) where T = x

Base.convert(::Type{TrackedReal{T}}, x::Real) where T = TrackedReal(convert(T, x))

Base.convert(::Type{TrackedReal{T}}, x::TrackedReal{S}) where {T,S} =
  error("Not implemented: convert tracked $S to tracked $T")

Base.:(<)(x::TrackedReal, y::TrackedReal) = data(x) < data(y)
Base.:(==)(x::TrackedReal, y::TrackedReal) = data(x) == data(y)

Base.eps(x::TrackedReal) = eps(data(x))

for f in :[isinf, isnan, isfinite].args
  @eval Base.$f(x::TrackedReal) = Base.$f(data(x))
end

Base.Printf.fix_dec(x::TrackedReal, n::Int) = Base.Printf.fix_dec(data(x), n)

Base.promote_rule(::Type{TrackedReal{S}},::Type{T}) where {S,T} =
  TrackedReal{promote_type(S,T)}

using DiffRules, SpecialFunctions, NaNMath

for (M, f, arity) in DiffRules.diffrules()
  arity == 1 || continue
  @eval begin
    @grad $M.$f(a::Real) =
      $M.$f(data(a)), Δ -> (Δ * $(DiffRules.diffrule(M, f, :a)),)
    $M.$f(a::TrackedReal) = track($M.$f, a)
  end
end

for (M, f, arity) in DiffRules.diffrules()
  arity == 2 || continue
  da, db = DiffRules.diffrule(M, f, :a, :b)
  f = :($M.$f)
  @eval begin
    @grad $f(a::Real, b::Real) = $f(data(a), data(b)), Δ -> (Δ * $da, Δ * $db)
    $f(a::TrackedReal, b::TrackedReal)  = track($f, a, b)
    $f(a::TrackedReal, b::Real) = track($f, a, b)
    $f(a::Real, b::TrackedReal) = track($f, a, b)
  end
end

# Eliminating ambiguity
import Base:^

^(a::TrackedReal, b::Integer) = track(^, a, b)

# Tuples

struct TrackedTuple{T<:Tuple}
  data::T
  tracker::Tracked{T}
end

data(xs::TrackedTuple) = xs.data
tracker(xs::TrackedTuple) = xs.tracker

accum!(x::Tuple, Δ::Tuple) = accum!.(x, Δ)
init_grad(x::Tuple) = init_grad.(x)
zero_grad!(x::Tuple) = zero_grad!.(x)

track(f::Call, xs::Tuple) = TrackedTuple(xs, Tracked{typeof(xs)}(f, zero.(xs)))

function Base.show(io::IO, xs::TrackedTuple)
  show(io, data(xs))
  print(io, " (tracked)")
end

Base.length(x::TrackedTuple) = length(data(x))

Base.getindex(xs::TrackedTuple, i::Integer) = track(getindex, xs, i)

@grad function getindex(xs::TrackedTuple, i)
  data(xs)[i], Δ -> (ntuple(j -> i == j ? Δ : 0, length(xs)), nothing)
end

# Array collection

function collect(xs)
  xs = Base.collect(xs)
  track(Call(collect, (tracker.(xs),)), data.(xs))
end

function scan(c::Call{typeof(collect)})
  foreach(scan, c.args[1])
end

function back_(c::Call{typeof(collect)}, Δ)
  foreach(back, c.args[1], data(Δ))
end
