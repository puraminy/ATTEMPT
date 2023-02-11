import numpy as np
from torch.distributions.relaxed_bernoulli import RelaxedBernoulli
def an(temperature, step, router, amin=1., arate=0.001):
     temperature = max(amin, temperature * np.exp(-arate * step))
     router = RelaxedBernoulli(temperature=temperature, 
            logits=router).rsample()            
     return router 

