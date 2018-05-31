export Target, targettrain!, difftargettrain!;

mutable struct Target{F, S, L}
	f::F
	dual_f::S
	loss::L
	σ::Real
	in::Array
	out::TrackedArray
	regulariser::Function
end

treelike(Target);

function Target(f, dual_f, loss; σ::Real = 1e-3, regulariser = regcov(0.5))::Target
	return Target(f, dual_f, loss, σ, Array{Float32, 0}(), TrackedArray(Array{Float32, 0}()), regulariser);
end

function (a::Target)(x)
	a.in = data(x);
	a.out = a.f(a.in);
	return a.out;
end

function Base.show(io::IO, a::Target)
	print(io, "Target(");
	print(io, a.f);
	print(io, ", ");
	print(io, a.dual_f);
	print(io, ")");
end

# targetprop

function targetprop!(a::Target, target)
	back!(a.loss(target, a.out)); # TODO: Regularisation
	ϵ = a.σ * randn(size(a.in))
	back!(a.loss(a.dual_f(data(a.f(a.in .+ ϵ))), a.in .+ ϵ))
	return data(a.dual_f(data(target)));
end

function targetprop!(a::Chain, target)
	foldl((m, x) -> targetprop!(x, m), target, reverse(a.layers))
	return target;
end

function targettrain!(model, modelloss, data, opt; η::Real = 0.001, cb = () -> ())
	cb = Optimise.runall(cb);
	opt = Optimise.runall(opt);
	@progress for d in data
		y_hat = Flux.data(model(d[1]));
		grad = param(y_hat);
		back!(modelloss(grad, d[2]));
		target = @fix y_hat - η * length(d[2]) * grad.grad;
		Optimise.@interrupts targetprop!(model, target);
		opt();
		cb() == :stop && break;
	end
end

# difftargetprop

function difftargetprop!(a::Target, packedTarget)
	(target, last) = packedTarget
	if !last
		target += data(a.out);
	end
	back!(a.loss(target, a.out)); # TODO: Regularisation
	back!(a.loss(a.dual_f(data(a.out)), a.in))
	back!(a.loss(a.dual_f(data(a.f(a.in .+ ϵ))), a.in .+ ϵ))
	nextTarget = data(a.dual_f(data(target)) - a.dual_f(data(a.out)));
	return (nextTarget, false);
end

function difftargetprop!(a::Chain, target)
	foldl((m, x) -> difftargetprop!(x, m), target, reverse(a.layers))
	return target;
end

function difftargettrain!(model, modelloss, data, opt; η::Real = 0.001, cb = () -> ())
	cb = Optimise.runall(cb);
	opt = Optimise.runall(opt);
	@progress for d in data
		y_hat = Flux.data(model(d[1]));
		grad = param(y_hat);
		back!(modelloss(grad, d[2]));
		target = @fix y_hat - η * length(d[2]) * grad.grad;
		Optimise.@interrupts difftargetprop!(model, (target, true));
		opt();
		cb() == :stop && break;
	end
end
