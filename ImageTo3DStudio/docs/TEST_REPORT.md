# Test report

Measured on 2026-11-11. The automated tests ran at commit `c433409`; the
measured runs used the same engine (the last change after them only affects
decimations where no attempt fits the budget, which none of the eight cases
needed). Every number below comes from `tools/quality_report.py` (stored in
`docs/quality_report.json`) or from the pytest run; nothing is estimated by
hand.

## Summary

- **8 of 8 end-to-end runs** produced complete, valid packages. Every
  single-image input was classified automatically (no category given).
- **78 automated tests passed**, 0 failed, 0 skipped, in 13 minutes,
  with the Blender and rbx-dom checks switched on.
- Every FBX was re-imported with Blender's own importer: triangle counts match
  and the embedded textures are linked to the material.
- Every RBXM and RBXMX was decoded with rbx-dom; binary and XML give identical
  trees.
- Every delivered mesh is watertight and passed validation.
- Not tested here: importing into Roblox Studio, and the bot against the live
  Discord API (see Known issues).

## Environment

| | |
|---|---|
| Machine | Linux container, 4 CPU cores, 15 GB RAM, no GPU |
| Python | 3.13.18 |
| Libraries | numpy 2.5.4, OpenCV 4.14 (headless), SciPy 1.18.1, scikit-image 0.26.0, trimesh 5.2.0, fast-simplification 0.2.0, mapbox-earcut 2.1.0, Pillow 13.0.0, discord.py 2.7.1, aiohttp 3.14.5, pytest 9.2.0 |
| FBX check | Blender 5.2.3 (`bpy` from PyPI) re-imports every FBX with the official importer |
| RBXM check | rbx-dom (`rbx_binary` 3.0.1, `rbx_xml` 3.0.1, `rbx_dom_weak` 4.2.0) decodes every `.rbxm` and `.rbxmx`; both trees must be identical |
| Test inputs | one real photo (the bed from the original request) and synthetic Blender Cycles renders, see `assets/test_images/SOURCES.md` |

## Measured runs

| Case | Input | Chosen strategy | Triangles (budget) | Size W x H x D (studs) | Maps | Time | Peak RAM | Silhouette IoU | Reduction impact (p95) | Blender FBX | rbx-dom RBXM/RBXMX |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Bed | bed_photo.jpg | template: bed | 4,512 (5,000) | 4.44 x 4.15 x 7.54 | AO, basecolor, normal, roughness | 225 s | 930 MB | 0.67 | 0.65 % (low) | 4512 tris, 3 packed images | 24 parts, identical: yes |
| Chair | chair_render.png | template: chair | 3,900 (4,000) | 1.41 x 3.21 x 1.71 | AO, basecolor, normal, roughness | 125 s | 930 MB | 0.64 | 0.24 % (negligible) | 3900 tris, 3 packed images | 13 parts, identical: yes |
| Table | table_render.png | template: table | 972 (2,000) | 5.00 x 2.68 x 2.86 | AO, basecolor, normal, roughness | 96 s | 930 MB | 0.83 | 0.24 % (negligible) | 972 tris, 3 packed images | 9 parts, identical: yes |
| Vase | vase_render.png | revolve | 2,852 (3,000) | 1.31 x 2.14 x 1.31 | AO, basecolor, roughness | 90 s | 930 MB | 0.97 | 0.53 % (negligible) | 2852 tris, 2 packed images | 6 parts, identical: yes |
| Plush | plush_render.png | inflate | 5,000 (5,000) | 1.89 x 2.67 x 1.08 | AO, basecolor, normal, roughness | 124 s | 930 MB | 0.92 | 1.98 % (moderate) | 5000 tris, 3 packed images | 20 parts, identical: yes |
| Emblem | emblem_drawing.png | extrude | 1,960 (2,000) | 3.76 x 3.56 x 0.45 | AO, basecolor, roughness | 102 s | 930 MB | 0.99 | 0.37 % (negligible) | 1960 tris, 2 packed images | 1 parts, identical: yes |
| Mug | mug_front.png + 3 more | visual_hull | 4,000 (4,000) | 8.47 x 5.93 x 5.92 | AO, basecolor, normal, roughness | 51 s | 1035 MB | 0.98 | 0.64 % (low) | 4000 tris, 3 packed images | 23 parts, identical: yes |
| ChairScan | chair_turn_00.jpg + 7 more | turntable | 4,000 (4,000) | 3.03 x 5.49 x 2.87 | AO, basecolor, normal, roughness | 53 s | 976 MB | 0.69 | 5.74 % (high) | 4000 tris, 3 packed images | 12 parts, identical: yes |

| Case | Watertight | Validation | Warnings | Template fit components |
|---|---|---|---|---|
| Bed | yes | ok | none | background_overlap 0.111, object_missed 0.128, probable_missed 0.214, edge_chamfer 0.422, edge_coverage 0.295 |
| Chair | yes | ok | none | background_overlap 0.060, object_missed 0.053, probable_missed 0.362, edge_chamfer 0.298, edge_coverage 0.238 |
| Table | yes | ok | none | background_overlap 0.013, object_missed 0.045, probable_missed 0.000, edge_chamfer 0.814, edge_coverage 0.057 |
| Vase | yes | ok | none | - |
| Plush | yes | ok | none | - |
| Emblem | yes | ok | none | - |
| Mug | yes | ok | none | - |
| ChairScan | yes | ok | none | - |

### Per-case notes

- **Bed** (the owner's photo): classified as a bed against sofa and cabinet;
  the fitted template covers 87 % of the certain bed pixels and
  11 % of its silhouette falls on background (along the top of the
  headboard and in the gaps between its slats). Frame measured as wood
  with a photo exemplar, bedding as fabric. Size 4.44 x 4.15 x 7.54 studs. The
  hidden side and the back of the headboard are mirrored estimates.
- **Chair** (render in a room): classified as a chair; three back slats
  counted from the photo, seat measured as fabric and frame as wood. The
  render's turned legs became square tapered legs: the template offers
  straight and round legs, not a turned profile.
- **Table**: classified as a table. The next level of detail (bevelled edges
  on every board) needs 2,700 triangles, more than the 2,000 budget, so the
  model stays at 972 triangles; the package's budget report says so.
- **Vase**: chosen automatically as a surface of revolution; silhouette IoU
  0.97.
- **Plush** (PNG with alpha): chosen automatically as an inflated silhouette;
  thickness is an estimate (the depth of a single photo is unknown).
- **Emblem** (flat drawing): chosen automatically as an extrusion; the centre
  hole is kept; IoU 0.99.
- **Mug** (front, side, back, top): visual hull; the handle opening is carved
  through, the inside of the cup is closed (no silhouette shows it); IoU
  0.98.
- **ChairScan** (8 unlabelled photos, little texture): feature matching found
  no reliable pairs, so the camera steps came from silhouette coherence:
  45.0 degrees per photo (true value 45) at 15 degrees elevation (true 14).
  Legs, seat, back frame and the three back slats are carved. The IoU is
  lower because the 4,000-triangle version of thin slats deviates from the
  photo; the reduction impact rating says the same (5.7 % of the size at the
  95th percentile). Use a budget of 8,000+ for objects with thin parts.

Silhouette IoU compares the delivered model, rendered with the recovered
camera, with the object in the first photo (for templates: with the pixels
that are certainly object). Templates score lower by design: they are
parametric approximations, and bedding or cushions that drape beyond the
frame are not modelled.

## Automated tests

| File | Tests | What they check |
|---|---|---|
| `test_geometry.py` | 27 | closed, outward-wound primitives and templates; mirrored transforms; flat and smooth normals; template resizing; 20,000-triangle MeshPart split; pinched-sheet repair |
| `test_textures.py` | 6 | non-overlapping atlas islands; meaningful maps only; gutter fill; exemplar colour statistics; wood variation |
| `test_export.py` | 8 | FBX round trip; FBX import in Blender with linked textures; RBXM/RBXMX structure and identical rbx-dom trees; valid glTF; OBJ/STL round trip; ZIP safety; re-export from a stored version; RBXMX float32 values |
| `test_pipeline.py` | 13 | bed photo end to end; package completeness and budget; edit without re-analysis; unreachable budget; revolve and inflate builds; automatic choice for emblem, vase and plush; four-view visual hull; photo-set calibration and thin legs; large image downscaling; invalid image |
| `test_security.py` | 12 | upload validation; decompression bombs; path traversal; signed links; download server; worker environment without secrets; token redaction in logs |
| `test_storage_jobs.py` | 4 | versions and ownership; restart recovery; concurrent jobs with isolated failures; timeout and cancel |
| `test_bot.py` | 5 | slash-command definitions; persistent buttons; modal and embed limits; icons |
| `test_bot_flow.py` | 3 | create, progress, result, edit, export and access control with stand-in Discord objects; non-image upload refused; busy user refused |

## Problems found by testing and fixed

Each of these was visible in a measured run or failed a test, and is covered
by a test or by the measured runs now.

| Problem | Fix |
|---|---|
| FBX reader stopped at Blender's 13-byte node sentinel | Sentinel handled, round-trip test |
| Material seeds differed between runs (`hash()` is salted) | Stable CRC32 seeds |
| Photo exemplars showed tiling seams | Vertex-centred footprints inside the exemplar |
| JPEG block edges and floor shadows leaked into the cut-out | Bilateral filtering, percentile Canny thresholds, floor-only shadow model, three-level zones |
| Chair and table were inflated or classified as a sofa | Shortlist of refined fits, segmentation fixes |
| Vase and plush were matched to the lamp or car template; the emblem was revolved | Drawing score, row convexity, lathe and organic acceptance rules |
| Visual hull from front/side/back gave a square mug | Top view added; top-view axis sign fixed |
| Carved and inflated surfaces had voxel steps and vertical bands | Continuous signed-distance fields, anti-aliased silhouettes, gradient normals |
| Preview renders had shadow-acne stripes | Normal offset and slope-scaled shadow bias |
| Baked AO and normal maps had banding and speckle | Slope-scaled AO bias, inverse-distance 8-NN transfer, per-part texture-space filtering |
| Photo-set mode failed on plain objects (0 of 7 photo pairs matched) | Silhouette-coherence calibration |
| Chair legs vanished in the photo-set model | Full-resolution field meshing, 128-cell grid, lighter smoothing |
| Chair-back gaps and the mug handle hole were filled | See-through gaps kept when they show the background colour |
| Extruded emblem had 36 open edges | Ring cleaning before ear-cutting, bevel validation |
| Validation deleted zero-area slivers and opened the surface | Slivers are removed by edge flips |
| RBXM and RBXMX decoded to different trees (float rounding) | XML writes the exact float32 values |
| The same photo gave slightly different cut-outs depending on what ran earlier | OpenCV's random generator is seeded before graph cuts and k-means |
| Photo-set steps landed 0.5 degrees off (flat coherence peak), carving away the chair-back slats | Fine stage on sharper masks, weighted centre of the near-best steps, full-circle snap |
| The cut-out swallowed thin slats into a see-through gap, which then opened completely | Only background-coloured pixels of a gap are opened |
| Decimation pinched thin slats into edges shared by four triangles | Gentler retries and pinch splitting; budget fallback keeps the smallest attempt |
| Blender probe crashed (script named like a stdlib module) | Renamed probe script |

## Known issues and limitations

- **One photo is one view.** Proportions, colours and materials of the
  visible side are measured; the back, the inside and anything hidden are
  estimated from symmetry and from what the object type usually looks like.
- **Object types.** Ten parametric templates (bed, chair, table, sofa,
  cabinet, shelf, lamp, house, car, box). Other objects become a surface of
  revolution, an extrusion, an inflated silhouette or (with several photos) a
  carved hull. Only bed, chair and table were fitted to test photos; the other
  templates are geometry-tested only.
- **Classification** is rule-based and was checked on 6 single-image inputs;
  it can pick the wrong type on unusual photos. `/3d edit` > *Re-analyse photo*
  or the `category` option fixes that.
- **Silhouette methods** cannot see concavities: the mug's inside and the
  space under a chair seat are closed.
- **Photo sets** of plain objects rely on roughly equal steps between photos.
  Thin parts need a larger triangle budget: at 4,000 triangles the photo-set
  chair deviates up to 6 % of its size from the full-resolution surface.
- **Mip-mapped viewers** (three.js, Roblox) can show faint lines where UV
  islands meet on sharp rims; the bake itself is seamless at full resolution.
- **Speed and memory.** A single-photo job takes 1.5-4 minutes on 4 cores
  (the bed photo 225 s), mostly spent comparing templates; multi-view jobs
  take under a minute. Peak memory is about 1 GB per job.
- **Not tested here:** importing into Roblox Studio (Studio does not run on
  Linux; the files follow Roblox's documented importer and were checked with
  Blender and rbx-dom instead) and running against the live Discord API (no
  bot token in the build environment; all bot flows were driven with
  stand-in Discord objects).
- **Test photos:** one real photo (the bed); the rest are synthetic renders,
  which are cleaner than phone photos.

## Reproduce

```bash
pip install -r requirements.txt
pytest -q                                     # automated tests
RBXCHECK=/path/to/rbxcheck BPY_PYTHON=/path/to/bpy-venv/bin/python pytest -q
python tools/quality_report.py --blender /path/to/bpy-venv/bin/python --rbxcheck /path/to/rbxcheck
```

`tools/quality_report.py` writes `docs/quality_report.json` (all numbers in
the tables above) and the sample packages in `samples/`.

## Update 2026-11-11: free-form objects

Reported failure: a picture of a decorated dessert egg came out once as a
bookshelf (template fit accepted with score 0.63) and once as a long prism
(the photo-set mode carved a volume from repeated photos).

| Check | Before | After |
|---|---|---|
| Egg photo, automatic strategy | `template` (shelf) | `inflate`, no template tried (outline 0 % straight) |
| Egg photo uploaded twice | photo-set carving, prism | repeat ignored, same result as one photo |
| Egg model | flat board with shelves | rounded solid 1.65 x 2.68 x 1.69 studs, 5,000 triangles, watertight, 0 validation warnings |
| Outline pixels showing the backdrop | brown and grey band around the rim | none (wrap projection, colours extended inwards) |
| Analysis time for the egg | 43.8 s (template fitting) | 1.0 s |
| Whole job (analysis, 1024 px textures, all exports) | not finished correctly | 30 s |

Test suite: 85 passed, 1 skipped (the Blender cross-check, Blender is not
installed on this machine), 9 min 45 s. New tests are in
`tests/test_sculpt.py`.

Not checked: import of the new egg package into Roblox Studio, and free-form
photos other than the egg and the plush render. Known limits of this mode: the
back is a mirror of the front (a face appears on both sides), a photo taken at
an angle is treated as a straight-on view, and parts that overlap in the photo
are not separated in depth.
