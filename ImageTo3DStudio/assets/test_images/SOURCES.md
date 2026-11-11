# Test images

| File | Source | Used for |
|---|---|---|
| `bed_photo.jpg` | Photo supplied by the project owner in the original request (product-style photo of a single bed). | Single-image template reconstruction (bed). |
| `egg_photo.jpg` | Picture supplied by the project owner (decorated dessert egg on a grey backdrop with a cast shadow, three-quarter view). The first version of the engine built a bookshelf from it. | Free-form reconstruction, template gate, shadow removal, repeated-photo guard. |
| `chair_render.png` | Synthetic render made with Blender 5.2 Cycles by `tools/make_test_images.py` (turned legs, upholstered seat, slatted back, room set). | Template classification and fitting (chair). |
| `table_render.png` | Synthetic render, same script (oak table on a studio backdrop). | Template fitting (table). |
| `vase_render.png` | Synthetic render, same script (glazed lathe-turned vase). | Surface-of-revolution strategy. |
| `plush_render.png` | Synthetic render with transparent background, same script (remeshed sphere plush toy). | Alpha-matte segmentation and silhouette inflation. |
| `mug_front.png`, `mug_side.png`, `mug_back.png`, `mug_top.png` | Synthetic orthographic renders, same script (mug with handle; front, side, back and top view). | Multi-view visual hull. |
| `chair_turn_00.jpg` ... `chair_turn_07.jpg` | Synthetic perspective renders, same script: the chair photographed every 45 degrees around it at 14 degrees elevation, saved as 640 px JPEG. The surfaces carry little texture, so feature matching fails and the silhouette-coherence fallback is what gets tested. | Photo-set mode (camera steps recovered from silhouette coherence when features do not match). |
| `emblem_drawing.png` | Flat artwork drawn with Pillow by `tools/make_drawing.py` (outlined gold star with a centre hole). | Drawing detection and the extrusion strategy with a hole. |

The synthetic objects are modelled in the render script independently of the
engine's templates, so tests do not grade the engine on its own geometry. They
are not real photographs: real photos add noise, clutter and lens effects that
these renders do not cover.
