# Test images

| File | Source | Used for |
|---|---|---|
| `bed_photo.jpg` | Photo supplied by the project owner in the original request (product-style photo of a single bed). | Single-image template reconstruction (bed). |
| `chair_render.png` | Synthetic render made with Blender 5.2 Cycles by `tools/make_test_images.py` (turned legs, upholstered seat, slatted back, room set). | Template classification and fitting (chair). |
| `table_render.png` | Synthetic render, same script (oak table on a studio backdrop). | Template fitting (table). |
| `vase_render.png` | Synthetic render, same script (glazed lathe-turned vase). | Surface-of-revolution strategy. |
| `plush_render.png` | Synthetic render with transparent background, same script (remeshed sphere plush toy). | Alpha-matte segmentation and silhouette inflation. |
| `mug_front.png`, `mug_side.png`, `mug_back.png` | Synthetic orthographic renders, same script (mug with handle). | Multi-view visual hull. |

The synthetic objects are modelled in the render script independently of the
engine's templates, so tests do not grade the engine on its own geometry. They
are not real photographs: real photos add noise, clutter and lens effects that
these renders do not cover.
