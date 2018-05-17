export TargetDense, targettrain!;

struct TargetDense{F, S, T, L}
	W::S
	dual_W::S
	b::T
	dual_b::T
	σ::F
	loss::L
end

function TargetDense(in::Integer, out::Integer, σ, loss; initW = glorot_uniform, initb = zeros)::TargetDense
	return TargetDense(initW(out, in), initW(in, out), initb(out), initb(in), σ, loss)
end

function target!(a::Chain, target)
	println("YIPEE");
end

function (a::TargetDense)(x)
	W, b, σ = a.W, a.b, a.σ
	@fix σ.(W*x .+ b)
end

function Base.show(io::IO, l::TargetDense)
	print(io, "TargetDense(", size(l.W, 2), ", ", size(l.W, 1))
	l.σ == identity || print(io, ", ", l.σ)
	print(io, ")")
end

function targettrain!(model, loss, data, opt; cb = () -> ())
  cb = Optimise.runall(cb)
  opt = Optimise.runall(opt)
  @progress for d in data
	Optimise.@interrupts target!(model, loss(d...))
    opt()
    cb() == :stop && break
  end
end
