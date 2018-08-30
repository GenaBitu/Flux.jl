export Target, targettrain!, difftargettrain!;

mutable struct Target{F, S, L}
	f::F
	dual_f::S
	loss::L
	σ::Real
	in::Array
	out::Union{Array, TrackedArray}
	debuglog::Dict{String, Array}
end

treelike(Target);

function Target(f, dual_f, loss; σ::Real = 0.2)::Target
	return Target(f, dual_f, loss, σ, Array{Float32, 0}(), TrackedArray(Array{Float32, 0}()), Dict{String, Array}());
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

function targetprop!(a::Target, targetTuple; debug::Array = [])
	(target, realgrad) = targetTuple;
	function debuglog(name, value)
		if name in debug
			if !haskey(a.debuglog, name)
				a.debuglog[name] = Array{typeof(data(value)), 1}();
			end
			push!(a.debuglog[name], data(value));
		end
	end

	if "difference" in debug
		inc = similar(a.in, (size(a.in, 1), size(a.in, 2) - 1));
		for i in 1:size(inc,2)
			inc[:,i] = a.in[:, i + 1] - a.in[:, i];
		end
		ind = mapreduce(i->inc[:, i], +, 1:size(inc,2))
		debuglog("difference", norm(a.f(ind)));
	end
	if "average" in debug
		avg = mean(map(i->a.in[:, i], 1:size(a.in, 2)));
		debuglog("average", mean(map(i->norm(data(a.f(a.in[:, i] - avg))), 1:size(a.in, 2))));
	end

	retgrad = [];
	if "angle" in debug
		paramin = param(a.in);
		fcopy = deepcopy(a.f);
		back!(fcopy(paramin), realgrad);
		retgrad = paramin.grad;
	end
	if isa(a.out, TrackedArray)
		l1 = a.loss(target, a.out); # TODO: Regularisation
		debuglog("Layer-local loss function", l1);
		back!(l1);
	end
	ϵ = a.σ * randn(size(a.in));
	l2 = a.loss(a.dual_f(data(a.f(a.in .+ ϵ))), a.in .+ ϵ);
	#l2 = a.loss(a.dual_f(a.f(a.in .+ ϵ)), a.in .+ ϵ); # Non-standard approach
	debuglog("Dual layer-local loss function", l2);
	if "Reverse auto-encoder loss" in debug
		l2i = a.loss(data(a.f(data(a.dual_f(data(a.out))))), data(a.out));
		debuglog("Reverse auto-encoder loss", l2i);
	end
	back!(l2);
	if isa(a.out,TrackedArray)
		veclength(a) = sqrt(sum(a.^2));
		vecangle(a, b) = begin
			acosd(min(dot(a, b) / (veclength(a) * veclength(b)), 1.0))
		end
		if "angle" in debug
			debuglog("angle", vecangle(vcat(map(i->vec(i.grad), params(a.f))...), vcat(map(i->vec(i.grad), params(fcopy))...)));
		end
		debuglog("jacobian", jacobian(a.f, a.in[:, rand(1:size(a.in, 2))]));
	end
	return (data(a.dual_f(data(target))), retgrad);
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
		Optimise.@interrupts targetprop!(model, (target, grad.grad); debug = debug);
		opt();
		cb() == :stop && break;
	end
end

# difftargetprop

function difftargetprop!(a::Target, packedTarget; debug::Array = [])
	(target, realgrad, last) = packedTarget
	if !last
		target += data(a.out);
	end
	(nextTarget, nextGrad) = targetprop!(a, (target, realgrad); debug = debug);
	nextTarget -= data(a.dual_f(data(a.out)));
	return (nextTarget, nextGrad, false);
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
		Optimise.@interrupts difftargetprop!(model, (target, grad.grad, true); debug = debug);
		opt();
		cb() == :stop && break;
	end
end
