"This module extends NormalizingFlowFilters with functionality from Random."
module RandomExt

using NormalizingFlowFilters: NormalizingFlowFilters
using Random

"""
    greeting()

Call [`NormalizingFlowFilters.greeting`](@ref) with a random name.


# Examples

```jldoctest
julia> @test true;

```

"""
NormalizingFlowFilters.greeting() = NormalizingFlowFilters.greeting(rand(5))

end
