![](https://i.imgur.com/iywjz8s.png)


# Collaborative Document

2024-11-26 Parallel Python Workshop, Day 2

Welcome to The Workshop Collaborative Document.

This Document is synchronized as you type, so that everyone viewing this page sees the same text. This allows you to collaborate seamlessly on documents.

----------------------------------------------------------------------------


This is the Document for today: https://edu.nl/9t8yq

Collaborative Document day 1: https://edu.nl/44h9b

Collaborative Document day 2: https://edu.nl/9t8yq

# **edu.nl/9t8yq**

##  🫱🏽‍🫲🏻 Code of Conduct

Participants are expected to follow these guidelines:
* Use welcoming and inclusive language.
* Be respectful of different viewpoints and experiences.
* Gracefully accept constructive criticism.
* Focus on what is best for the community.
* Show courtesy and respect towards other community members.

## ⚖️ License

All content is publicly available under the Creative Commons Attribution License: [creativecommons.org/licenses/by/4.0/](https://creativecommons.org/licenses/by/4.0/).

## 🙋Getting help

To ask a question, just raise your hand.

If you need help from a helper, place a pink post-it note on your laptop lid. A helper will come to assist you as soon as possible.

## 🖥 Workshop website

[link](https://esciencecenter-digital-skills.github.io/2024-11-26-ds-parallel/)

## 👩‍🏫👩‍💻🎓 Instructors

Francesco Nattino, Leon Oostrum, Benjamin Czaja, Johan Hidding

## 👩‍💻👩‍💼👨‍🔬🧑‍🔬🧑‍🚀🧙‍♂️🔧 Roll Call
Name/ pronouns (optional) / job, role / social media (twitter, github, ...) / background or interests (optional) / city

...

## 🗓️ Agenda
|  Time | Topic                          |
| -----:|:------------------------------ |
| 09:30 | Welcome, icebreaker and recap  |
| 09:45 | Using Numba                    |
| 10:30 | Coffee break                   |
| 10:45 | Delayed evaluation             |
| 12:30 | Lunch                          |
| 13:30 | Computing fractals in parallel |
| 14:45 | Coffee break                   |
| 15:00 | (exercise continued)           |
| 15:45 | Tea break                      |
| 16:30 | Presentations of group work    |
| 16:45 | Post-workshop Survey           |
| 17:00 | Drinks                         |

## 🏢 Location logistics
* Coffee and toilets are in the hallway, just outside of the classroom.
* If you leave the building,
  be sure to be accompanied by someone from the escience center to let you back in through the groundfloor door
* For access to this floor you might need to ring the doorbell so someone can let you in
* In case of an emergency, you can exit our floor using the main staircase.
  Or follow green light signs at the ceiling to the emergency staircase.
* **Wifi**: Eduroam should work. Otherwise use the 'matrixbuilding' network, password should be printed out and available somewhere in the room.

## 🎓 Certificate of attendance
If you attend the full workshop you can request a certificate of attendance by emailing to training@esciencecenter.nl.
Please request your certificate within 8 months after the workshop, as we will delete all personal identifyable information after this period.

## 🔧 Exercises

### Challenge: Numbify calc_pi

Create a Numba version of `calc_pi`. Time it.

```python=
import random

def calc_pi(N):
    M = 0
    for i in range(N):
        # Simulate impact coordinates
        x = random.uniform(-1, 1)
        y = random.uniform(-1, 1)

        # True if impact happens inside the circle
        if x**2 + y**2 < 1.0:
            M += 1
    return 4 * M / N
```

**Solution**

```python=
import random

@numba.jit
def calc_pi_numba(N):
    M = 0
    for i in range(N):
        # Simulate impact coordinates
        x = random.uniform(-1, 1)
        y = random.uniform(-1, 1)

        # True if impact happens inside the circle
        if x**2 + y**2 < 1.0:
            M += 1
    return 4 * M / N
```

Native (vanilla) Python:
```python=
%%timeit
calc_pi(10**6)  # ~1 sec
```
With Numba:
```python=
%%timeit
calc_pi(10**6)  # ~10 ms
```

### Challenge: Run the workflow

Given this workflow:

```python
x_p = add(1, 2)
y_p = add(x_p, 3)
z_p = add(x_p, -3)
```

Visualize and compute `y_p` and `z_p` separately, how often is x_p evaluated?
Now change the workflow:

```python
x_p = add(1, 2)
y_p = add(x_p, 3)
z_p = add(x_p, y_p)
z_p.visualize(rankdir="LR")
```

We pass the yet uncomputed promise `x_p` to both `y_p` and `z_p`. Now, only compute `z_p`, how often do you expect `x_p` to be evaluated? Run the workflow to check your answer.

**Solution**

```python=
y_p.compute()
```
Runs the two additions.

```python=
z_p.compute()
```
Also runs the two additions. So `x_p` is computed twice. Thus: delay as much as possible the call to `.compute()`!


### Challenge: Understand `gather`

Can you describe what the `gather` function does in terms of lists and promises? hint: Suppose I have a list of promises, what does `gather` allow me to do?

**Solution**

Suppose I have the following list:

```python
work = [add(1, 2), add(2, 3), add(3, 4)]
result = gather(*work)
```

`gather` turns a list of delayeds into a delayed list. If we visualize it, we get something like:

![](https://esciencecenter-digital-skills.github.io/parallel-python-workbench/fig/dask-gather-example.svg)

### Challenge: Design a `mean` function and calculate $/pi$

Write a delayed function that computes the mean of its arguments. Use it to esimates pi several times and returns the mean of the results.

```python
mean(1, 2, 3, 4).compute()  # 2.5
```

Make sure that the entire computation is contained in a single promise.

**Solution**

```python
@delayed
def mean(*args):
    return sum(args) / len(args)
```

Test the function:

```python=
def test_mean():
    assert mean(1, 2, 3, 4).compute() == 2.5

test_mean()  # no errors - it works!
```

Let's now use `mean` to compute pi!

```python
import random

@numba.jit(nogil=True)
def calc_pi_numba(N):
    M = 0
    for i in range(N):
        x = random.uniform(-1, 1)
        y = random.uniform(-1, 1)
        if x**2 + y**2 < 1.0:
            M += 1
    return 4 * M / N
```

Let's run this in parallel:

```python=
work = mean(*(delayed(calc_pi_numba)(10**7) for _ in range(16)))
```

Let's time the single run:
```python=
%%timeit
calc_pi_numba(10**7)  # ~ 90 ms
```

```python=
%%timeit
work.compute()  # ~ 370 ms
```

With 4 cores this is close to ideal speedup!


### Challenge: Generate all even numbers

Can you write a generator that generates all even numbers? Try to reuse `integers()`.

(Extra: Can you generate the Fibonacci numbers?)

**Solution**

```python=
def even_integers():
    a = 1
    while True:
        yield 2*a
        a += 1
```

Or reusing `integers`:

```python
(i for i in integers() if i % 2 == 0)
```

### Challenge: Line numbers

Change `printer` to add line numbers to the output.

**Solution**

```python=
def printer():
    a = 1
    while True:
        x = yield
        print(f"{a}: {x}")
        a +=1
```

### Challenge: Gather multiple outcomes

We’ve seen that we can gather multiple coroutines using `asyncio.gather`. Now gather several `calc_pi` computations, and time them.

**Solution**

```python=
async with timer() as t:
    work = [asyncio.to_thread(calc_pi_numba, 10**7) for _ in range(16)]
    result = await asyncio.gather(*work)
    print(result)
```

## 🧠 Collaborative Notes

Jupyter Hub: https://jupyter.snellius.surf.nl/jhssrf016/

### Numba

Almost "free" performance - or at least at a small cost.

Let's sum a range of numbers:

```python
sum(range(1, 10**7))
```

The most naive implementation would be:

```python
def sum_range(a: int):
    x = 0
    for i in range(a):
        x += i
    return x
```

Let's timeit:
```python=
%%timeit
sum_range(10**7)  # ~650 ms
```

With numba:

```python=
import numba

@numba.jit  # only add this decorator!
def sum_range(a: int):
    x = 0
    for i in range(a):
        x += i
    return x
```

Now  timing:

```python=
%%timeit
sum_range(10**7)  # ~250 ns
```

We get an amazing speedup! What's happening? Normal Python is interpreted: interpreter goes through code line by line, which makes it slow. Numba compile the code to machine code (JIT = Just in Time compilation)- to this is executed directly. The drawback: compiled version of your code is much more "rigid" (fixed data types, only limited types understood).

With Numpy:

```python=
import numpy as np
np.arange(10**7).sum()  # 15.3 ms
```

It is much slower than the numba version, since Numpy is allocating a large array in memory and then summing all its elements. The numba implementation does not need to do that!

Numba compiles the code the first time it is executed!


[Exercise: Challenge Numbify calc_pi](#Challenge-Numbify-calc_pi)

Can we combine Numba implementation with parallel execution?

Let's do it with multiple threads!

```python
import threading
```

```python
%%timeit
t1 = threading.Thread(target=calc_pi_numba, args=(10**7,))
t2 = threading.Thread(target=calc_pi_numba, args=(10**7,))

t1.start()
t2.start()

t1.join()
t2.join()
```
One run ~100 ms, two runs ~200 ms. We have the Global Interpreter Lock (GIL) which prevents speedup using threads!

Let's modify the Numba decorator:

```python
@numba.jit(nogil=True)
def calc_pi_numba(N):
    ...
```

The GIL is released and we get ideal scaling, using threads!

Equivalent way to do the same thing as the decorator (so we don't have to modify the original function):

```python=
calc_pi_numba = numba.jit(calc_pi, nogil=True)
```

### Delayed evaluation

Another dask interface: Dask delayed. More suitable for functional programming.

```python
from dask import delayed

@delayed
def add(a, b):
    result = a + b
    print(f"{a} + {b} = {result}")
    return result
```
Let's build a first "promise" (`_p`):

```python=
x_p = add(1, 1)
```

`x_p` is a delayed object! Function has not run yet! Run it with:

```python
x_p.compute()
```

We can combine delayed and visualize task graph:

```python=
x_p = add(1, 2)
y_p = add(x_p, 3)
z_p = add(x_p, y_p)
z_p.visualize(rankdir="LR")
```

![](https://esciencecenter-digital-skills.github.io/parallel-python-workbench/fig/dask-workflow-example.svg)

[**Exercise: Run the workflow**](#Challenge-Run-the-workflow)

**BREAK until 10:45**

Let's turn `calc_pi` into a delayed function - we can do it also using `delayed` as a function, not as a decorator:

```python=
x_p = delayed(calc_pi)(N)
x_p.compute()
```

Let's define a function that add an arbitrary amounts of numbers:

```python
def add(*args):
    return sum(args)
```

so we can do:

```python
numbers = [1, 2, 3, 4]
add(*numbers)  # 10
```

Let's define the following function:

```python
@delayed
def gather(*args):
    return list(args)
```

[**Exercise: Understand `gather`**](#Challenge-Understand-gather-)

[**Exercise: Design a mean function and calculate pi**](#Challenge-Design-a-mean-function-and-calculate-pi)

Additional speedup with Numba: add `parallel = True` to decorator and switch from using `range` in for loops to `prange` (parallel range). Numba will automatically parallelize your code.

**BREAK until 11:45**

### AsyncIO

AsyncIO builds on the concept of coroutines.

```python
def integers():
    a = 1
    while True:
        yield a  # yield gives control to caller
        a += 1
```

Let's loop over the sequence using the generator built above:

```python=
for i in integers():
    print(i)
    if i > 10:
        break
```

To deal with infinite sequences (iterators), we can use `itertools.islice`:

```python=
from itertools import islice
list(islice(integers(), 0, 10))
```

`islice` also works lazily (similarly to `map` seen yesterday).

[**EXERCISE: Generate all even numbers**](#Challenge-Generate-all-even-numbers)

Other use of `yield`:

```python
def printer():
    while True:
        x = yield
        print(x)
```

```python=
p = printer()
next(p)  # get the iterator ready to receive values
p.send("Mercury")
p.send("Venus")
p.send("Earth")
```

Asyncio is part of the standard library and adds two elements to the syntax: `async` (modify function or loop to make it asynchronous) and `await` (to run async content).

```python=
import asyncio

async def coutner(name):
    for i in range(5):
        print(f"{name:<10} {i:03}")
        await asyncio.sleep(0.2)
```

`await` can only appear inside or in front of `async` content.

```python=
await counter("Venus")
```

Join two async counters:

```python
await asyncio.gather(counter("Earth"), counter("Moon"))
```

If you want to run this in a Python script, this is how it should look like:

```python=
import asyncio

async def main():
    ...

if __name__ == "__main__":
    asyncio.run(main())
```

Asynchronous functions can only be called by asynchronous functions (or by something like `asyncio.run`).

To time asyncio material copy this snippet:

```python
from dataclasses import dataclass
from typing import Optional
from time import perf_counter
from contextlib import asynccontextmanager


@dataclass
class Elapsed:
    time: Optional[float] = None


@asynccontextmanager
async def timer():
    e = Elapsed()
    t = perf_counter()
    yield e
    e.time = perf_counter() - t
```

Use this (and test it) with:

```python
async with timer() as t:
    await asyncio.sleep(0.2)
print(f"that took {t.time} seconds")
```

To send work to different threads, need to use:

```python=
async with timer() as t:
    await asyncio.to_thread(calc_pi_numba, 10**7)
```

Check time with:

```python=
t.time
```

[**EXERCISE: Gather multiple outcomes**](#Challenge-Gather-multiple-outcomes)


**LUNCH BREAK - back at 13:30**

### Mandelbrot fractal

Consider the recursive relation (starting from $z_1 = 0$):

$$z_{n+1} = z_n + c$$

If the series does not diverge, $c$ is part of the Mandelbrot set. For instance, $c = -1$ is part of the set. $c = 1$ is not.

Useful to know: if the absolute value of $z_n$ becomes larger than 2, the sequence will diverge!

Serial implementation:

```python=
max_iter = 256
width = 256
height = 256
center = -0.8+0.0j
extent = 3.0+3.0j

scale = max((extent / width).real, (extent / height).imag)

result = np.zeros((height, width), int)

for j in range(height):
    for i in range(width):
        c = center + (i - width // 2 + (j - height // 2)*1j) * scale
        z = 0
        for k in range(max_iter):
            z = z**2 + c
            if (z * z.conjugate()).real > 4.0:
                break
        result[j, i] = k
```

Visualize with:

```python=
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
plot_extent = (width + 1j * height) * scale
z1 = center - plot_extent / 2
z2 = z1 + plot_extent
ax.imshow(result**(1/3), origin='lower', extent=(z1.real, z2.real, z1.imag, z2.imag))
ax.set_xlabel("$\Re(c)$")
ax.set_ylabel("$\Im(c)$")
```

Try changing to different region:

```python=
max_iter = 1024
center = -1.1195+0.2718j
extent = 0.005+0.005j
```

#### Hints

It can help to create a `BoundingBox` class through which you manage your domain:

```python=
from dataclasses import dataclass
from dask import array as da
import numba

@dataclass
class BoundingBox:
    origin: complex
    scale: float
    size: tuple[int, int]

    @staticmethod
    def centered(center: complex, scale: float, size: tuple[int, int]):
        origin = center - size[0] * scale/2 - size[1] * scale/2 * 1j
        return BoundingBox(origin, scale, size)

    @property
    def real_axis(self):
        return np.arange(self.size[0]) * self.scale + self.origin.real

    @property
    def imag_axis(self):
        return (np.arange(self.size[1]) * self.scale + self.origin.imag)

    @property
    def extent(self):
        return (self.origin.real, self.origin.real + self.size[0]*self.scale,
                self.origin.imag, self.origin.imag + self.size[1]*self.scale)

    def dask_grid(self, chunks=(128, 128)):
        x = da.arange(self.size[0], chunks=chunks[0]) * self.scale + self.origin.real
        y = (da.arange(self.size[1], chunks=chunks[1]) * self.scale + self.origin.imag) * 1j
        return x[None,:] + y[:,None]

    def grid(self):
        return self.real_axis[None,:] + self.imag_axis[:,None] * 1j

@numba.jit(nopython=True, nogil=True)
def orbit(c: complex128) -> Generator[complex128, None, None]:
    z = 0.0+0.0j
    while True:
        yield z
        z = z**2 + c

from numba import int32, complex128

@numba.jit(nogil=True)
def escape_time(maxit: int32, c: complex128) -> int32:
    for i, z in enumerate(orbit(c)):
        if i >= maxit or (z*z.conjugate()).real >= 4.0:
            return i

@numba.vectorize([int32(int32, complex128)], nopython=True)
def v_escape_time(maxit, c):
    return escape_time(maxit, c)
```


## 📚 Resources

* [Numba](https://numba.pydata.org/)
* [Julia](https://julialang.org/): Alternative, fully JIT-compiled language
* [Mojo](https://www.modular.com/mojo): Another new type-annotated language, similar syntax to Python (actually, superset of it)
* [Lesson material](https://esciencecenter-digital-skills.github.io/parallel-python-workbench/)
