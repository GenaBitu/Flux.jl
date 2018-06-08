export Target, targettrain!, difftargettrain!;

mutable struct Target{F, S, L}
	f::F
	dual_f::S
	loss::L
	σ::Real
	in::Array
	out::Union{Array, TrackedArray}
end

treelike(Target);

function Target(f, dual_f, loss; σ::Real = 0.2)::Target
	return Target(f, dual_f, loss, σ, Array{Float32, 0}(), TrackedArray(Array{Float32, 0}()));
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

function targetprop!(a::Target, target; debug::Array = [])
	function debugprint(name, value)
		if name in debug
			print(name);
			print(": ");
			println(data(value));
		end
	end

	if isa(a.out,TrackedArray)
		l1 = a.loss(target, a.out); # TODO: Regularisation
		debugprint("l1", l1)
		back!(l1);
	end
	ϵ = a.σ * randn(size(a.in));
	l2 = a.loss(a.dual_f(data(a.f(a.in .+ ϵ))), a.in .+ ϵ); # Should be this, but doesn't work for some reason...
	#l2 = a.loss(a.dual_f(a.f(a.in .+ ϵ)), a.in .+ ϵ);
	if "l2" in debug
		l2noiseless = a.loss(a.dual_f(data(a.f(a.in))), a.in); # Should be this, but doesn't work for some reason...
		debugprint("l2", l2noiseless)
	end
	if "l2i" in debug
		l2i = a.loss(data(a.f(data(a.dual_f(data(a.out))))), data(a.out));
		debugprint("l2i", l2i)
	end
	back!(l2);
	return data(a.dual_f(data(target)));
end

function targetprop!(a::Chain, target; debug::Array = [])
	foldl((m, x) -> targetprop!(x, m; debug = debug), target, reverse(a.layers))
	return target;
end

function targettrain!(model, modelloss, data, opt; η::Real = 0.001, cb = () -> (), debug::Array = [])
	cb = Optimise.runall(cb);
	opt = Optimise.runall(opt);
	@progress for d in data
		y_hat = Flux.data(model(d[1]));
		grad = param(y_hat);
		back!(modelloss(grad, d[2]));
		target = @fix y_hat - η * length(d[2]) * grad.grad;
		if length(debug) > 0
			println("Iteration:");
		end
		Optimise.@interrupts targetprop!(model, target; debug = debug);
		opt();
		cb() == :stop && break;
	end
end

# difftargetprop

function difftargetprop!(a::Target, packedTarget; debug::Array = [])
	(target, last) = packedTarget
	if !last
		target += data(a.out);
	end
	nextTarget = targetprop!(a, target; debug = debug) - data(a.dual_f(data(a.out)));
	return (nextTarget, false);
end

function difftargetprop!(a::Chain, target; debug::Array = [])
	foldl((m, x) -> difftargetprop!(x, m; debug = debug), target, reverse(a.layers))
	return target;
end

function difftargettrain!(model, modelloss, data, opt; η::Real = 0.001, cb = () -> (), debug::Array = [])
	cb = Optimise.runall(cb);
	opt = Optimise.runall(opt);
	@progress for d in data
		y_hat = Flux.data(model(d[1]));
		grad = param(y_hat);
		back!(modelloss(grad, d[2]));
		target = @fix y_hat - η * length(d[2]) * grad.grad;
		if length(debug) > 0
			println("Iteration:");
		end
		Optimise.@interrupts difftargetprop!(model, (target, true); debug = debug);
		opt();
		cb() == :stop && break;
	end
end
