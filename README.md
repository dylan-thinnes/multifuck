# multifuck

Brainfuck, with extensions as I think of them.

So far:

- Multithreading
- Multidimensional memory

This interpreter is slightly different from the original brainfuck spec:

- Numbers are 32-bit signed integers, meaning they can become negative. Looping
  `[` and `]` check that the value is 0, if it is less than 0 or greater than
  0, they loop.
- Only the left bracket, `[`, runs the check for a conditional jump. The right
  bracket `]` always makes an unconditional jump back to its matching `[`. The
  original spec did the check on both brackets.

# Multidimensional memory

Multidimensional memory adds two instructions, `*` and `/`, which allows the
memory pointer to access new "dimensions" on the tape.

All pointers start out in dim 0, where `<` and `>` move along the x-axis in
memory. The `*` reorients the pointer to move along dim 1, at which point `>`
and `<` move along the y-axis. A second use of `*` would make the dim 2, at
which point we'd move along the z-axis, and so-on to dim 4, dim 5, etc.

While the `*` instruction can be said to reorient from dim n to dim n+1, the
`/` instruction reorients from dim n to dim n-1. This means that negative
dimensions exist in our memory, and are just as orthogonal as all the positive
dimensions are.

# Multithreading

Multithreading adds one instruction, `&`, which spawns a thread which waits one
cycle before continuing in the same manner on the program tape.

This means that a thread that is just spawned will initially run the same
instructions as its parent, just with a one cycle delay. There are a number of
ways to get threads to diverge so that they no longer run the same
instructions, some examples of which are in `./examples`.

All threads read from and write to the same memory on the same cycle, so
simultaneous increments and decrements will either stack or cancel out, rather
than overwrite one another. Similarly, thread that checks to loop will not see
the increments made to a position in memory by another thread until the next
cycle.

One easy way to get threads to diverge is with the following pattern:

```
&+-[
  PROGRAM 1
]

PROGRAM 2
```

In this program, the parent thread will run `PROGRAM 1` then `PROGRAM 2`,
whereas the spawned thread will only run `PROGRAM 2`.

In order to get full divergence where both threads run completely different
programs, the following pattern is useful:

```
&+-[+-
  PROGRAM 1
]
[
  PROGRAM 2
]
```

In this program, the parent thread will run `PROGRAM 1`, and the spawned thread
will run `PROGRAM 2`. Total divergence!

Common programs such as `[->+<]` ("move value one cell right") can be well
accelerated by running multiple threads in tandem. Many fun examples about
spawning many threads with certain delays, moving values more quickly, spawning
n threads, can be found in `./examples`.
