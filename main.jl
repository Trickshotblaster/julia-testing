using Flux: gradient
print("hello world")

thing = rand(Float32, 1, 10)
global w1 = rand(Float32, 10, 2)
global expected = zeros(Float32, 1, 2)
loss_fn(thing, w1, expected) = sum(expected - (thing * w1)) ^ 2
print("\n")
print(gradient(loss_fn, thing, w1, expected)[2]) # w1 grad

for step in 0:100
    noise = rand(Float32, 1, 10)
    grads = gradient(loss_fn, noise, w1, expected)
    w1_grad = grads[2]
    global w1 -= w1_grad * 0.05
end
print("\n")
print(loss_fn(rand(Float32, 1, 10), w1, expected))