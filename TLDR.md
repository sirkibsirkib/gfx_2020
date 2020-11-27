# TL;DR for myself

## Graphics pipeline

Your application provides: a list of triangles, with verts indexing a given array [X; N].

Step 1 (vertex shader step) computes [Y; N],
where X is your user-defined structure, with more or less any contents at all,
and where Y:
1. includes special variable `vec4 gl_position`
2. boils down to floating-point vector types `vec2`, `vec3` etc.

Step 2 (fragment shader step) ultimately populates a framebuffer,
essentially a pixel value per screen pixel for one frame.
Data does not move neatly through a chain of vertex->frag shader compute steps;
rather, the gpu will execute your frag shader given Y'=interpolate([Y0, Y1, Y2], gl_position)
for every fragment whose Y' represents the linear interpolation of corresponding 
fields from Y0, Y1 and Y2 _inside_ the triangle with verts Y0.gl_position, Y1.gl_position, Y2.gl_position.
