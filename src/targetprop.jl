export TargetDense, TargetSoftmax, targettrain!, difftargettrain!;

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
	regulariser::Function
	λ::Number
end

treelike(TargetDense);

function TargetDense(in::Integer, out::Integer, σ, loss; initW = glorot_uniform, initb = zeros, regulariser = regcov, λ::Number = 0.5)::TargetDense
	return TargetDense(param(initW(out, in)), param(initW(in, out)), param(initb(out)), param(initb(in)), σ, loss, Array{Float32, 0}(), TrackedArray(Array{Float32, 0}()), regulariser, λ);
end

function (a::TargetDense)(x)
	W, b, σ = a.W, a.b, a.σ;
	a.in = data(x);
	a.out = @fix σ.(W*a.in .+ b);
	a.out = a.out .+ (a.λ .* a.regulariser(a.out));
	return a.out;
end

l2(x) = mean(x.^2);
l1(x) = mean(abs.(x));
noise(x) = rand(size(x));
function regcov(x);
	xx = x .- mean(x,2);
	return mean(xx*transpose(xx));
end
function logcov(x)
	o = x*transpose(x);
	return mean(o) - mean(log.(diag(o) .+ 1f-4 ));
end
#Generalization in Deep Learning, Kenji Kawaguchi, Leslie Pack Kaelbling, Yoshua Bengio
function darc1(x)
	l = sum(abs.(x),2)/size(x,2);
	return l[indmax(Flux.Tracker.data(l))];
end

function Base.show(io::IO, l::TargetDense)
	print(io, "TargetDense(", size(l.W, 2), ", ", size(l.W, 1));
	l.σ == identity || print(io, ", ", l.σ);
	print(io, ")");
end

# targetprop

function targetprop!(a::TargetDense, target)
	back!(a.loss(target, a.out));
	W, b, σ = a.dual_W, a.dual_b, a.σ;
	dual(x) = σ.(W*data(x) .+ b);
	back!(a.loss(dual(a.out), a.in))
	return data(dual(target));
end

function targetprop!(a::Chain, target)
	map(x->target=targetprop!(x, target), reverse(a.layers));
	return target;
end

function targettrain!(model, modelloss, data, opt; cb = () -> ())
	η = 0.001; # TODO: Get from optimiser
	cb = Optimise.runall(cb);
	opt = Optimise.runall(opt);
	@progress for d in data
		y_hat = Flux.data(model(d[1]));
		grad = param(y_hat);
		back!(modelloss(grad, d[2]));
		target = y_hat - η * grad.grad;
		Optimise.@interrupts targetprop!(model, target);
		opt();
		cb() == :stop && break;
	end
end

# difftargetprop

function difftargetprop!(a::TargetDense, target)
	target += data(a.out);
	back!(a.loss(target, a.out));
	W, b, σ = a.dual_W, a.dual_b, a.σ;
	dual(x) = σ.(W*data(x) .+ b);
	back!(a.loss(dual(a.out), a.in))
	return data(dual(target) - dual(a.out));
end

function difftargetprop!(a::Chain, target)
	map(x->target=difftargetprop!(x, target), reverse(a.layers));
	return target;
end

function difftargettrain!(model, modelloss, data, opt; cb = () -> ())
	η = 0.001; # TODO: Get from optimiser
	cb = Optimise.runall(cb);
	opt = Optimise.runall(opt);
	@progress for d in data
		y_hat = Flux.data(model(d[1]));
		grad = param(y_hat);
		back!(modelloss(grad, d[2]));
		target = y_hat - η * grad.grad;
		Optimise.@interrupts difftargetprop!(model, target);
		opt();
		cb() == :stop && break;
	end
end
