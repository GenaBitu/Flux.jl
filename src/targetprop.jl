export TargetDense, TargetSoftmax, targettrain!;

# TargetDense
mutable struct TargetDense{F, S, T, L}
	W::S
	dual_W::S
	b::T
	dual_b::T
	σ::F
	loss::L
	out::TrackedArray
end

function TargetDense(in::Integer, out::Integer, σ, loss; initW = glorot_uniform, initb = zeros)::TargetDense
	return TargetDense(param(initW(out, in)), param(initW(in, out)), param(initb(out)), param(initb(in)), σ, loss, TrackedArray(Array{Float32, 0}()));
end

function (a::TargetDense)(x)
	W, b, σ = a.W, a.b, a.σ;
	a.out = @fix σ.(W*data(x) .+ b);
	return a.out;
end

function Base.show(io::IO, l::TargetDense)
	print(io, "TargetDense(", size(l.W, 2), ", ", size(l.W, 1));
	l.σ == identity || print(io, ", ", l.σ);
	print(io, ")");
end

# TargetSoftmax

mutable struct TargetSoftmax{S, T, L}
	dual_W::S
	dual_b::T
	loss::L
	out::TrackedArray
end

function TargetSoftmax(dim::Integer, loss; initW = glorot_uniform, initb = zeros):TargetSoftmax
	return TargetSoftmax(initW(dim, dim), initb(dim), loss, TrackedArray(Array{Float32, 0}()));
end

function (a::TargetSoftmax)(x)
	a.out = softmax(x);
	return a.out;
end

# targetprop

function target!(a::TargetSoftmax, target)
	println("TargetSoftmax");
	back!(a.loss(a.out, target));
	W, b, σ = a.dual_W, a.dual_b, identity;
	return @fix σ.(W*data(a.out) .+ b);
end

function target!(a::TargetDense, target)
	println("TargetDense");
	back!(a.loss(a.out, target));
	W, b, σ = a.dual_W, a.dual_b, a.σ;
	return @fix σ.(W*data(a.out) .+ b);
end

function target!(a::Chain, target)
	println("Chain");
	map(x->target=target!(x, target), reverse(a.layers));
	return target;
end

function targettrain!(model, data, opt; cb = () -> ())
  cb = Optimise.runall(cb);
  opt = Optimise.runall(opt);
  @progress for d in data
	model(d[1]);
	Optimise.@interrupts target!(model, d[2]);
    opt();
    cb() == :stop && break;
  end
end
