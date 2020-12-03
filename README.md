# gfx_2020

Tinkering with gfx_hal, to get a feel for graphics programming in general, and Vulkan programming in particular.
For now, the aim is to build a very simply engine for 2D sprite rendering.

## Design goals:
1. Draw many many sprites as fast as possible
2. Handles GPU resources safely
3. Expose users to instance data, and minimize

## API in a nutshell
Between frames, users can:
1. load and unload texture images from the GPU
2. write instance data: (a) transform matrix, and (b) texture scissor

The drawing revolves around an operation, which draws a set of instances, given:
1. a loaded image to use as the texture (i.e. spritesheet)
2. a range into the instance transform buffer
3. a range into the instance texture scissor buffer

### Depth and Transparency
Alpha blending and depth testing is used in the rendering pipeline.
Instances drawn with semi-transparent pixels (alpha between 0 and 256) will NOT correctly occlude deeper instances.
To get around this, use the painters algorithm to draw semi-transparent instances _after_ the ones they occlude. 

## References
This work takes a lot from others' examples:
1. [gfx-hal's quad example](https://github.com/gfx-rs/gfx/tree/master/examples/quad)
2. [learn gfx-hal tutorial (a bit outdated)](https://rust-tutorials.github.io/learn-gfx-hal/)
