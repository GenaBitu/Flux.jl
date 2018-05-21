export TargetDense, TargetSoftmax, targettrain!;

# TargetDense
mutable struct TargetDense{F, S, T, L}
	W::S
	dual_W::S
	b::T
	dual_b::T
	σ::F
	loss::L
	in::Array
	out::TrackedArray
end

treelike(TargetDense);

function TargetDense(in::Integer, out::Integer, σ, loss; initW = glorot_uniform, initb = zeros)::TargetDense
	return TargetDense(param(initW(out, in)), param(initW(in, out)), param(initb(out)), param(initb(in)), σ, loss, Array{Float32, 0}(), TrackedArray(Array{Float32, 0}()));
end

function (a::TargetDense)(x)
	W, b, σ = a.W, a.b, a.σ;
	a.in = data(x);
	a.out = @fix σ.(W*a.in .+ b);
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
	in::Array
	out::Array
end

treelike(TargetSoftmax);

function TargetSoftmax(dim::Integer, loss; initW = glorot_uniform, initb = zeros):TargetSoftmax
	return TargetSoftmax(param(initW(dim, dim)), param(initb(dim)), loss, Array{Float32, 0}(),Array{Float32, 0}());
end

function (a::TargetSoftmax)(x)
	a.in = data(x);
	a.out = softmax(a.in);
	return a.out;
end

# targetprop

function target!(a::TargetSoftmax, target)
	W, b, σ = a.dual_W, a.dual_b, tanh;
	ret = @fix σ.(W*a.out .+ b);
	back!(a.loss(ret, a.in))
	return data(ret);
end

function target!(a::TargetDense, target)
	back!(a.loss(a.out, target));
	W, b, σ = a.dual_W, a.dual_b, a.σ;
	ret = @fix σ.(W*data(a.out) .+ b);
	back!(a.loss(ret, a.in))
	return data(ret);
end

function target!(a::Chain, target)
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
