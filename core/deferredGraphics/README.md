# Deferred Graphics

This is a deferred graphics modul. Render consists of two parts: 
* first subpass - render geomerty and save information in position (f32), normal (f32), albedo (u8) and emission (u8) maps with all necessary information about fragment material (according to PBR) in position's and normal's alpha channals.
* second subpass - render a screen space, and lighting every fragment according to geometry information from maps about this point.
In addition to the main render pipeline contains low cost abient lighting, and outlighting extension of base render. 

Disadvantages of the implementation:
* no anti-aliasing
* there is basically no rendering of transparent objects
* memory consumption for maps

For transparent objects rendering were added transparent layers, base render discards all fragments with color alpha component < 1.0. Fragments with color alpha component < 1.0 are rendered in several transparent layers, fragments are sorted by depth and also processed by deferred graphics pipeline. Base layer and transparent layers are combined in `layersCombiner`.

## Additional pipelines:
* Shadow maps generation. Must be prerended before lighting.
* Skybox map. Must be prerended before layers combining.
* Light scattering. Rendered for each light images in common result image. Result image must be rendered before layers combining.
* Bloom. There was used bloom image down sampling method. Must be renderd with layers combining result.
* Guassian blur. Must be renderd with layers combining result.
* BoundingBox.
* SSLR. Not finished yet.
* SSAO. Not finished yet.
* PostProcessing. Gets results of all stages for finalize frame processing.

## Render scheme
```
Begin
├── stage 1
│   ├── Copy info from objects to buffer
├───┘
├── stage 2
│   ├── Shadows
│   ├── Skybox
├───┘
├── stage 3
│   ├── Deferred Graphics
│   ├── Transparent Layers
├───┘
├── stage 4
│   ├── Scattering
├───┘
├── stage 5
│   ├── Layers Combining
├───┘
├── stage 6
│   ├── Selector
│   ├── SSLR
│   ├── SSAO
│   ├── Bloom
│   ├── Blur
│   ├── BoundingBox
│   ├── PostProcessing
├───┘
End
```

## Supported light sources:
* Spot light.

## Optimization of spot light rendering

A single vkCmdDraw is used for each light sources, in which every single pass draw inside surface of piramid, corresponded to projection and view matrixes of light source. Fragment shader works only with pyramid fragments of this light point. There is no shading of points outside of light volume by this way. Results of every light pass is blended in attachment using function of maximum. The result is image with lighted fragments of pyramids. The result is combined with ambient light pass. It fills fragments which outs of light pyramids. Eventually This method gives same image of deferred render, but calculations are more fast. Also differentiation of light sources calculations lets using various pipelines and shaders for every light source. It lets to render variose effects without performance drop. In fact performance depends on count of shaded fragments and do not depend on light sources count.
