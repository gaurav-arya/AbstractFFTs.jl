for f in (:fft, :bfft, :ifft, :rfft)
    pf = Symbol("plan_", f)
    @eval begin
        function ChainRulesCore.frule((_, Δx, _), ::typeof($f), x::AbstractArray, dims) 
            y = $f(x, dims)
            Δy = $f(Δx, dims)
            return y, Δy
        end
        function ChainRulesCore.rrule(::typeof($f), x::T, dims) where {T<:AbstractArray}
            y = $f(x, dims)
            project_x = ChainRulesCore.ProjectTo(x)
            ax = axes(x)
            function fft_pullback(ȳ)
                x̄ = project_x($pf(similar(T, ax), dims)' * ChainRulesCore.unthunk(ȳ))
                return ChainRulesCore.NoTangent(), x̄, ChainRulesCore.NoTangent()
            end
            return y, fft_pullback
        end
    end
end

for f in (:brfft, :irfft)
    pf = Symbol("plan_", f)
    @eval begin
        function ChainRulesCore.frule((_, Δx, _), ::typeof($f), x::AbstractArray, d::Int, dims) 
            y = $f(x, d::Int, dims)
            Δy = $f(Δx, d::Int, dims)
            return y, Δy
        end
        function ChainRulesCore.rrule(::typeof($f), x::T, d::Int, dims) where {T<:AbstractArray}
            y = $f(x, d, dims)
            project_x = ChainRulesCore.ProjectTo(x)
            ax = axes(x)
            function fft_pullback(ȳ)
                x̄ = project_x($pf(similar(T, ax), d, dims)' * ChainRulesCore.unthunk(ȳ))
                return ChainRulesCore.NoTangent(), x̄, ChainRulesCore.NoTangent(), ChainRulesCore.NoTangent()
            end
            return y, fft_pullback
        end
    end
end

# shift functions
function ChainRulesCore.frule((_, Δx, _), ::typeof(fftshift), x::AbstractArray, dims)
    y = fftshift(x, dims)
    Δy = fftshift(Δx, dims)
    return y, Δy
end
function ChainRulesCore.rrule(::typeof(fftshift), x::AbstractArray, dims)
    y = fftshift(x, dims)
    project_x = ChainRulesCore.ProjectTo(x)
    function fftshift_pullback(ȳ)
        x̄ = project_x(ifftshift(ChainRulesCore.unthunk(ȳ), dims))
        return ChainRulesCore.NoTangent(), x̄, ChainRulesCore.NoTangent()
    end
    return y, fftshift_pullback
end

function ChainRulesCore.frule((_, Δx, _), ::typeof(ifftshift), x::AbstractArray, dims)
    y = ifftshift(x, dims)
    Δy = ifftshift(Δx, dims)
    return y, Δy
end
function ChainRulesCore.rrule(::typeof(ifftshift), x::AbstractArray, dims)
    y = ifftshift(x, dims)
    project_x = ChainRulesCore.ProjectTo(x)
    function ifftshift_pullback(ȳ)
        x̄ = project_x(fftshift(ChainRulesCore.unthunk(ȳ), dims))
        return ChainRulesCore.NoTangent(), x̄, ChainRulesCore.NoTangent()
    end
    return y, ifftshift_pullback
end

# plans
function ChainRulesCore.frule((_, _, Δx), ::typeof(*), P::Plan, x::AbstractArray) 
    y = P * x 
    Δy = P * Δx
    return y, Δy
end
function ChainRulesCore.rrule(::typeof(*), P::Plan, x::AbstractArray)
    y = P * x
    project_x = ChainRulesCore.ProjectTo(x)
    function fft_pullback(ȳ)
        x̄ = project_x(P' * ȳ)
        return ChainRulesCore.NoTangent(), ChainRulesCore.NoTangent(), x̄
    end
    return y, fft_pullback
end
