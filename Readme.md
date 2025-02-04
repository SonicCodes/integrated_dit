is interesting idea, still on progress, dis just copy paste of past codes :333

So basically the idea is that instead of learning fixed trajectory from N(0,1) to target distribution , why don't we do soemthing close to integration (atleast) using block causal transformers, that way we can let the model loose abt the timestep and let it figure it out by it self, 

Implementation will basically be like, 10 different discrete points on the entire trajectory, the model sees different levels of noise and it'll predict what "v" needs to be predicted to reach the next denoising, so technically its autoregressive on the absolute sense...