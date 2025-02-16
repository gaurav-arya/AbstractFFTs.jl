module AbstractFFTs

export fft, ifft, bfft, fft!, ifft!, bfft!,
       plan_fft, plan_ifft, plan_bfft, plan_fft!, plan_ifft!, plan_bfft!,
       rfft, irfft, brfft, plan_rfft, plan_irfft, plan_brfft,
       fftdims, fftshift, ifftshift, fftshift!, ifftshift!, Frequencies, fftfreq, rfftfreq

include("definitions.jl")

if !isdefined(Base, :get_extension)
    include("../ext/AbstractFFTsChainRulesCoreExt.jl")
end

end # module
