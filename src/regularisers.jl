function l2(α)
	return x -> begin
		return α * sum(x.^2);
	end
end

function l1(α)
	return x -> begin
		return α * mean(abs.(x));
	end
end

function regcov(α)
	return x -> begin
		xx = x .- mean(x,2);
		return α * mean(xx*transpose(xx));
	end
end

function logcov(α)
	return x -> begin
		o = x*transpose(x);
		reg = mean(o) - mean(log.(diag(o) .+ 1f-4 ));
		return α * reg;
	end
end

#Generalization in Deep Learning, Kenji Kawaguchi, Leslie Pack Kaelbling, Yoshua Bengio
function darc1(α)
	return x -> begin
		l = sum(abs.(x),2)/size(x,2);
		reg = l[indmax(Flux.Tracker.data(l))];
		return α * reg;
	end
end
