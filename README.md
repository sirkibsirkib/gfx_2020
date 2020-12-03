# gfx_2020

Tinkering with gfx_hal, to get a feel for graphics programming in general, and Vulkan programming in particular.
For now, the aim is to build a very simply engine for 2D sprite rendering.

Design goals:
1. Draw many many sprites as fast as possible
2. Handles GPU resources safely
3. User manages instance data buffers themselves

This work takes a lot from others' examples:
1. [gfx-hal's quad example](https://github.com/gfx-rs/gfx/tree/master/examples/quad)
2. [learn gfx-hal tutorial (a bit outdated)](https://rust-tutorials.github.io/learn-gfx-hal/)
