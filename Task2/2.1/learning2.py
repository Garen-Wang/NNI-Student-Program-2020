import torch
import math


dtype = torch.float
device = torch.device('cpu')

x = torch.linspace(-math.pi, math.pi, 1000, dtype=dtype, device=device)
y = torch.sin(x)

a = torch.randn((), device=device, dtype=dtype, requires_grad=True)
b = torch.randn((), device=device, dtype=dtype, requires_grad=True)
c = torch.randn((), device=device, dtype=dtype, requires_grad=True)
d = torch.randn((), device=device, dtype=dtype, requires_grad=True)

eta = 1e-6
for epoch in range(2000):
    y_pred = a + b * x + c * x ** 2 + d * x ** 3
    loss = (y_pred - y).pow(2).sum()

    if epoch % 100 == 99:
        print(epoch, loss.item())

    loss.backward()

    with torch.no_grad():
        a -= eta * a.grad
        b -= eta * b.grad
        c -= eta * c.grad
        d -= eta * d.grad

        a.grad = None
        b.grad = None
        c.grad = None
        d.grad = None

    print(f'result: y = {a.item()} + {b.item()} x + {c.item()} x^2 + {d.item()} x^3')