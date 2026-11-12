# Technical documentation

## Architecture

```
Discord  <->  bot/ (discord.py)           commands.py  views.py  interactions.py  delivery.py  embeds.py  icons.py
                 |  StudioService: create / edit / restore / export / deliver
                 v
            storage/ (SQLite, WAL)        database.py  projects.py  jobs.py
                 |
            workers/queue.py  -- asyncio queue, N slots, timeouts, memory limit, cancel, restart recovery
                 |  spawns  python -m workers.processor job.json   (one process per job)
                 v
            engine/                        analyzer  fitting  reconstruction  procedural  geometry
                                           materials  textures  render  optimization  validation  pipeline
                 v
            exporters/                     fbx  rbxm  interchange (glb/obj/stl)  previews  packaging  convert  blender
```

The bot process never runs geometry code. A job is a JSON payload executed by
a separate Python process that streams `PROGRESS`/`RESULT`/`ERROR` lines on
stdout; a crash, an out-of-memory kill or a timeout only fails that job.

### Data flow of `/3d create`

1. Attachments are size-checked from Discord metadata, downloaded, then
   validated by their own bytes (magic number, full decode, pixel limit,
   decompression-bomb guard). Files are renamed `image_N.ext`; nothing uploaded
   is ever executed or extracted.
2. A project row, version 1 (with its `BuildSettings`) and a job row are
   created; the job is queued and the user sees a progress message.
3. The worker runs `pipeline.analyse_project` (image analysis +
   reconstruction, stored as `source/spec.json` + `spec_data.npz`) and
   `pipeline.build_version` (geometry at budget, textures, validation,
   export, previews, package, ZIP).
4. The queue marks the version ready; the bot posts the result card with the
   preview, the ZIP (if it fits the upload limit) and persistent buttons.

### Editing without re-analysis

`spec.json` is the editable intermediate representation: template name and
fitted parameters (or profile/outline/volume for generic strategies),
measured materials, fitted camera. Every edit creates a new version with new
`BuildSettings` (budget, size, scale, bevel, shading, materials, parameter
overrides, collision, pivot, grouping, texture options) and only reruns
`build_version`. Older versions keep their settings; their packages are kept
for the newest `KEEP_VERSIONS` and can be rebuilt from settings otherwise.
Restore = new version with an old version's settings (non-destructive).

## Image analysis (engine/analyzer.py)

- **Loading**: EXIF orientation, alpha detection, downscale to 1024 px
  working size, 35 mm focal length from EXIF when present.
- **Segmentation** (no learned models). Background colour model: k-means in
  CIELAB on a 3 % border band. Edges: Canny on bilateral-filtered lightness
  with thresholds from the gradient distribution, plus chroma edges. Floor
  removal: flood fill from the bottom border through floor-like chroma that
  does not cross edges. Shadow suppression: floor chromaticity at 25-88 % of
  its brightness. Output is a three-level zone map - certain background
  (border-connected, background-coloured, outside the object's hull), certain
  object (strongly distinct colour), undecided - which seeds OpenCV GrabCut for
  the final cut-out. Transparent PNGs use their alpha directly.
- **Features**: bounding box, aspect, fill ratio, solidity, holes, mirror
  symmetry (best vertical axis by IoU), rectangularity, line segments (LSD)
  by orientation, CIELAB palette, a "drawing-likeness" score (flat colour,
  dark outline, saturation) and a width profile.

## Busy pictures (engine/analyzer.py)

- **Letterbox bars**: flat near-black rows and columns at the edges (mean
  below 22, spread below 3) are cropped on loading, at most 30 % per axis.
- **Busy background**: when the cut-out covers more than 8 % of the picture's
  edge on a non-plain backdrop, a second graph cut seeded with a centred
  rectangle is tried; it replaces the first when it touches the edge less
  than half as much. Tested on five street and nature photos: the first cut
  took the trees or the hillside, the second the car, the elephant, the
  tiger and the lion.
- **MPO** (the JPEG variant many phones and cameras write) is accepted.

## Reconstruction (engine/reconstruction.py, engine/fitting.py, engine/sculpt.py, engine/surround.py)

### Strategy selection
Templates are only candidates when the outline has straight edges. The
outline is simplified to a polygon within 0.6 % of its bounding-box diagonal;
`outline_straightness` is the share of the perimeter in sides longer than 16 %
of that diagonal (a curve breaks into short sides, a straight edge survives).
Below 0.15 (`TEMPLATE_MIN_STRAIGHTNESS`) the object is free-form and no
template is fitted at all. Measured: egg photo 0.00, plush 0.00, bed photo
0.29, chair 0.65, table 0.74, emblem 0.90. Before this gate a box-shaped
template covered any compact blob well enough to be accepted (the decorated
egg scored 0.63 as a bookshelf).

Templates are also skipped when the cut-out runs along more than 25 % of the
picture's edge (it has swallowed the surroundings), and a silhouette with a
see-through opening only tries the templates that have gaps (chair, table,
shelf, bed). A refined fit is rejected when more than 16 % of the template
covers background (`TEMPLATE_MAX_SPILL`); the bed photo needs 11 %, the
chair 6 %, the table 1 %, and a photo of two cars that fitted a sofa 22 %.

Otherwise `rank_categories` scores categories from the features; the six best
templates are fitted coarsely. The two best coarse fits plus the best
feature-ranked template (or the third coarse fit) are refined with up to 170
evaluations each, and the lowest refined score wins if it is below the
acceptance score:

| Silhouette | Acceptance | Reason |
|---|---|---|
| lathe-like: mirror symmetry > 0.935 and > 95 % of rows are one run | 0.80 | a surface of revolution reproduces it exactly |
| organic: solidity > 0.8 and few straight lines | 0.85 | compact blobs are rarely furniture or vehicles |
| anything else | 1.12 (`TEMPLATE_ACCEPT_SCORE`) | |

The lamp template is not tried for lathe-like silhouettes (revolve is exact)
nor for silhouettes with many split rows (a lamp is a lathe shape). Without an
accepted template: a drawing score above 0.55 selects extrusion, a lathe-like
silhouette revolution, anything else inflation. The drawing score combines
the share of perfectly flat pixels (|Laplacian| < 3), the share of the four
most common colours (4 bits per channel) and a dark outline along the
silhouette. Every tested template and its score is written to the
reconstruction notes. The user can force a category or mode, and can
re-analyse later.

### Template fitting (analysis-by-synthesis)
A hypothesis is (template parameters, azimuth, elevation, field of view, 2D
scale and offset). It is built at the coarsest level of detail, rasterised as
a material label map by the software rasteriser and scored:

```
score = 1.3 * rendered pixels on certain background
      + 1.6 * certain-object pixels not covered
      + 0.4 * probable-object pixels not covered   (vertical envelope below outlined edges)
      + 0.9 * chamfer(rendered part boundaries -> image edges)
      + 0.45 * chamfer(image edges inside the object -> rendered boundaries)
      + small log-prior on deviation from real-world proportions
```

Search: a grid of 15 azimuths x 4 elevations x 2 fields of view with the 2D
placement solved from bounding boxes, then deterministic pattern search
(coordinate steps with halving) from the three best distinct viewpoints over
camera and all length parameters; the 2D placement is kept stable while 3D
parameters change. Repeated elements (bed and chair slats, drawer rows,
shelves, sofa cushions) are counted afterwards by sampling the photo along the
projected 3D line where they must be and counting runs of object colour or
seams.

### Materials from the photo
The fitted model is projected at full resolution; each material's pixels are
eroded away from boundaries. Pixels of all materials are clustered together in
CIELAB (low lightness weight so one surface in light and shadow stays one
cluster); clusters are owned by material families (wood vs fabric) and each
material takes its best allowed cluster. The colour is the mean of the
55-92 % (or 72-96 % for matte fabrics) luminance band in linear light, i.e. lit
diffuse colour without highlights and cast shadows. Contaminated or
low-agreement regions fall back to a confident material of the same family or
to the template default; same-chroma materials of one family share the lit
colour. The kind (wood, fabric, metal ...) is chosen from hue/saturation,
structure-tensor coherence (grain), texture strength and highlight share,
limited to the slot's plausible kinds. For wood, stone and leather, the
largest visible flat face is perspective-rectified into an exemplar patch.

### Generic strategies
- **Revolve**: per-row radius around the symmetry axis, smoothed, revolved.
- **Extrude**: outer contour and holes, Douglas-Peucker simplification (LOD
  controlled), ear-cut caps, mitred bevel ring. Points the ear-cutter would
  drop (collinear or repeated, in the ring or its inset) are removed first so
  caps and walls share every edge; a bevel whose inset ring would intersect
  itself or another ring is halved, and dropped after three tries.
- **Inflate** (`engine/sculpt.py`): the solid is the union of all balls whose
  equatorial disc fits inside the silhouette (Twarog, Tappen and Adelson
  2012, "Puffball"). With `D` the distance to the outline, the squared half
  thickness is `H2(p) = max_c D(c)^2 - |p - c|^2`, the lower envelope of
  parabolas, computed exactly in two passes (Felzenszwalb and Huttenlocher
  2012) on a grid of at most 288 px. A disc becomes a sphere, a bar a
  cylinder, and a part that is narrow in the photo stays equally thin in
  depth. The field `(H2 - z^2) / (2 sqrt(max H2))` is meshed like the other
  fields. The outline is first smoothed (Gaussian on the mask, re-thresholded
  at 0.5) to remove pixel stairs. If at least 30 % of the widest row touches
  the lowest 2 % of the object, the outline is mirrored below the ground line
  before inflating, so the model stands on a flat, full-depth base. The map
  `H2` is stored with the spec; an edit never recomputes it. Reconstructions
  saved before this model (no `inflate_model` in the spec) still build with
  the old `sqrt(d/dmax)` profile.
- **See-through openings** (inflate): holes in the silhouette that show the
  backdrop are kept (a mug handle, a ring). The hole test weighs lightness
  at 0.12 instead of 0.55, so a shadow on the backdrop inside the hole still
  counts as backdrop. A silhouette with such an opening is never revolved.
- **Synthesised back** (`engine/surround.py`, inflate and revolve): the
  surface is described per image row by an axis, a half width `a` (outline)
  and a half depth `b` (thickness map, or `a` for a lathe), and every point
  by its azimuth (0 = facing the camera, 180 = back). On the front the
  azimuth comes from the photo column (`sin(theta) = u`), towards the sides
  it changes over to the geometric angle around the row's ellipse, so it is
  continuous across the outline.
  1. Camera elevation: seen from above, a horizontal ring of the object lies
     lower in the photo at the front than at the sides, by
     `b * sin(elevation) * cos(theta)`. Thirteen candidates from 0 to 37
     degrees are tried; the one whose unwrapped strip has the straightest
     rows (lowest spread of lightness along each row) wins, and a level
     camera is kept unless a candidate is at least 4 % better. The egg photo
     gives 26.7 degrees.
  2. The front from -75 to 75 degrees is unwrapped along those rings into a
     strip, and its lighting (Gaussian-blurred brightness, sigma 1/10 of the
     strip) is divided out.
  3. Image quilting (Efros and Freeman 2001) fills 60 to 300 degrees at
     about 160 rows: patches of 28 px with 9 px overlap, from the strip and
     its mirror image, from rows within 3.5 % of the target row (with a
     quadratic penalty on the offset), joined by minimum-error boundary cuts.
     The bands 60-75 and 285-300 degrees are copied unchanged and the quilt
     must join them. Penalties: colour regions narrower than 55 % of the
     strip are details (k-means in CIELAB, 8 clusters, region width), a
     patch with more than 8 % detail pixels is excluded, and a source
     position already used nearby costs extra, so nothing is stamped twice
     in a row.
  4. The placement is replayed at full resolution, and the back is re-lit
     smoothly from the photo's lighting at one join band, through the row's
     median, to the other.
  Texels with an azimuth up to 64 degrees take the photo directly, beyond 74
  degrees the synthesised back, with a smooth blend between. The relief map
  is computed for both. If synthesis is not possible the mirrored back is
  used.
- **Wrap projection** (inflate): every texel of the solid, front or back,
  lies behind one object pixel of the orthographic photo, so the colour is
  looked up directly, without visibility or facing weights; nothing is left
  bare at the rim. Pixels within 1.2 % of the outline are a blend with the
  backdrop and are replaced by the nearest colour further inside
  (`extend_colours`). The band-passed lightness of the photo (Gaussian at
  size/400 minus Gaussian at size/45, soft-limited) is added to the height
  channel with an amplitude of 0.6 % of the object height: small bright
  details rise, small dark ones sink, in the normal map only.
- **Visual hull**: every orthographic view (front/side/back/left/top)
  contributes its silhouette's signed distance, low-pass filtered to the voxel
  spacing and scaled to studs; the hull is the minimum over views.
- **Carving silhouettes** (visual hull and photo sets): interior holes are
  filled, except holes whose pixels match the background colour model
  (median weighted CIELAB distance below 5) and that cover at least 0.2 % of
  the object: these are real gaps (between chair-back slats, inside a mug
  handle). Only their background-coloured pixels (distance below 8) are
  opened, so thin parts the cut-out missed inside a gap stay solid.
- **Determinism**: OpenCV's random generator is seeded before every graph
  cut and k-means, so the same photo always gives the same model.
- **Turntable** (unlabelled photos around the object): SIFT features, ratio
  test, essential matrix (RANSAC) and `recoverPose` between consecutive photos
  give the yaw steps. If fewer than half of the pairs match (plain surfaces),
  equal steps are assumed and the signed step (10-120 degrees) and the camera
  elevation (0-35 degrees) are chosen by silhouette coherence: carve a coarse
  36-cell volume with all silhouettes, reproject it into every photo and
  maximise the mean IoU with the silhouettes (Hernandez, Schmitt and Cipolla
  2007). A fine stage on sharper masks (64 cells, steps of 0.5 degrees) takes
  the coherence-weighted centre of all steps within 1 % of the best, because
  the peak is flat; a set whose steps add up to about 360 degrees snaps to
  360/n when that explains the silhouettes as well. The same score picks
  direction and elevation for SIFT-derived angles. Cameras share one model
  (distance from the object's pixel height, 2D shift from its bounding box).
  The final 128-cell volume takes, per voxel, the k-th largest signed distance
  over views (k = views - 1): the continuous form of "inside all but one
  silhouette". If no step explains the silhouettes (agreement below 55 %),
  the job falls back to single-image mode and says why.
- **Photo-set guards**: unlabelled photos that repeat an earlier one (same
  normalised silhouette, IoU above 0.95, and mean colour difference below 9)
  are dropped; fewer than four distinct unlabelled photos, or a recovered
  camera arc below 75 degrees, are not carved (two or three silhouettes only
  give a prism) and the first photo is reconstructed on its own.
- **Meshing of fields** (inflate, visual hull, turntable): Gaussian
  smoothing (0.9 voxel for reference views, 0.6 for photo sets, whose thin
  parts are only a few cells wide), always on the full grid, marching cubes
  at level 0, Taubin smoothing; shading
  normals come from the field gradient, so neither the voxel grid nor the
  marching-cubes facets print bands into the baked normal map. After
  decimation the low mesh takes its shading normals from the dense surface.

## Art styles (engine/stylize.py)

`BuildSettings.art_style` picks the look of a version; the analysis is shared,
so `/3d edit` > Art style switches between them without re-analysing.

| Style | What it does |
|---|---|
| `realistic` | Photo texture, synthesised back (default). |
| `cartoon` | Every photo of the projection (and the synthesised back) is turned into flat colours: broad lightness changes removed (Gaussian at 1/8 of the object, 60 %), mean-shift and bilateral smoothing, 8 colours by k-means in CIELAB (lightness weighted 0.45), near-duplicates merged, up to 3 accent colours for small, clearly different, saturated or very dark/light regions, median-filtered labels, colours saturated by 18 %. Procedural materials become plastic of their own colour. No relief map, occlusion in the colour at most 15 %. |
| `blocky` | The hand-built block look of many Roblox games (no studs). See below. |

`blocky`:
1. The photo's light and shadow are flattened, the realistic model is baked
   once at 512 px and its texels are kept as colour samples.
2. The solid is voxelized (trimesh) with 16 cubes along its longest side,
   larger if the triangle budget needs it (never fewer than 10). Cells are
   centred on multiples of the cube size measured from the body's mirror
   plane, so mirrored cells line up; the body is then made exactly symmetric
   (union with its mirror image; accessories are not mirrored). Edge-only
   contacts between cubes are filled, so the surface stays manifold.
3. Each outer cube takes the median colour of the 160 texel samples under its
   face, unless a much lighter or darker feature covers 15 % of it (an eye is
   one cube). The body and every accessory get their own palette (6 and 2
   colours, near-duplicates merged at ΔE 20, accents for the body),
   practically identical colours across regions (ΔE 12) are unified, a
   majority filter over 26 neighbours removes noise but never merges clearly
   different colours (ΔE 42), and if at least 70 % of mirrored body cubes
   already agree, the rest are matched (the rarer colour wins).
4. Same-coloured cubes are grouped into slabs of up to 4 x 1 x 4 cubes, with
   the order of the horizontal axes alternating per layer so seams stagger.
   The texture darkens the 7 % of each slab next to its border (a groove)
   and lifts the band just inside it; the normal map lowers the border, which
   reads as a bevel.
5. Eyes: two small (at most 6 cubes) very dark spots on the front, level,
   mirror images of each other, at least 55 % surrounded by the main body
   colour, in the upper two thirds. The upper-left cube of each gets a large
   and a small white highlight square.
6. No occlusion is baked. The `.rbxm` holds one coloured Part per slab
   (up to 600).

## Shape: body and accessories (engine/sculpt.py `shape_parts`)

Before inflation the silhouette is split: the body is what survives an
opening with a disc of 18 % of the smaller side; components of the rest
larger than 0.4 % of the silhouette are accessories (a cone, ears, a handle).
If the body is under 45 % of the silhouette the object is made of thin parts
and stays whole. A body whose mirror image overlaps it by at least 86 % is
replaced by the average of its signed distance field and the mirrored one
(exactly symmetric, same size), its axis becomes the model's X = 0 plane,
accessories keep their place, small gaps created between them and the body
are bridged, and enclosed gaps that were not see-through holes in the photo
are filled. The regions are stored with the spec, so styles can colour
accessories separately.

## Procedural modelling (engine/procedural.py, engine/geometry.py)

Components: bevelled/rounded boards (`box` with per-face grids and analytic
rounded normals), cushions (subdivided rounded boxes with domed faces),
turned rods and lathes (profile revolution with hard-corner splitting),
extruded polygons with holes and bevels, lofts between cross-sections.
Every component is closed, outward-wound and carries physical UVs (1 UV unit
= 1 stud) with the U axis along the wood grain, plus a Roblox-native
primitive (Block, Cylinder, Ball) used for the `.rbxm`.

Templates: bed, chair, table, sofa, cabinet, shelf, lamp, house, car and box -
each a function of named physical parameters (studs, counts, toggles) with
ranges and real-world defaults.

Level of detail: one quality value `q` controls bevel segments, round
segments and cushion subdivision; each can also be raised separately.

## Optimisation (engine/optimization.py)

1. Binary search on `q` for the highest detail under the budget, then each
   sub-level is raised separately so the count lands close to the request.
2. If even `q = 0` is too heavy (or for carved meshes), the budget is split
   across parts by `sqrt(triangles) * sqrt(area)` with a 12-triangle floor and
   each part is decimated with quadric edge collapse; UVs are re-charted.
   When a collapse pinches a thin part (an edge shared by more than two
   triangles), gentler settings are retried; any pinch that remains is split
   into separate sheets (corners joined only across two-triangle edges, the
   copies moved 2 mm apart), so the result stays closed and manifold.
3. Unreachable budgets are reported with the minimum safe count.
4. Quality impact: symmetric surface deviation between the full-detail and the
   delivered model (40k samples each way, KD-trees), reported as mean and 95th
   percentile in % of the bounding diagonal, rated negligible/low/moderate/high.
   Validation repairs zero-area sliver triangles by flipping their longest
   edge with the neighbouring triangle (deleting them would open the surface)
   and removes index-degenerate and duplicate triangles.
5. Roblox: meshes above 20,000 triangles are split into several MeshParts;
   pivot placement (bottom centre by default, centre, back, origin); size in
   studs; collision proxy (box, convex hull or per-part hulls) and the
   recommended CollisionFidelity.

## Textures (engine/textures.py)

1. **Atlas**: islands = connected faces in the physical UV layout; one texel
   density for the whole model, chosen by binary search so a shelf packer fits
   every island with padding into one square atlas; islands are rotated to
   landscape.
2. **Texels**: all triangles are rasterised in UV space; each texel gets its
   3D position, normal, tangent frame, part and physical UV.
3. **Materials** per texel: solid 3D wood (distorted ring field around an
   off-board axis + fibres + pores), fabric weave with optional quilting,
   brushed metal, Worley leather, fBm stone, plastics; all tinted to the
   measured colour and keeping its mean. Photo exemplars are synthesised with
   histogram-preserving triangle-grid tiling (Heitz and Neyret 2018) in
   decorrelated, Gaussianised colour space; offsets keep each tile's footprint
   inside the exemplar so non-tileable photo patches never show seams.
4. **Photo projection** (generic strategies, box): texels are projected into
   the source camera(s), tested against a depth map rendered from that camera,
   weighted by facing angle (4th power across views) and mirrored across the
   symmetry plane for hidden sides; remaining texels use a photo-matched
   material colour.
5. **Ambient occlusion**: 24-96 orthographic depth maps from Fibonacci-sphere
   directions; occlusion = cosine-weighted visible fraction, with the floor as
   an occluder for downward directions. The depth test uses a slope-scaled
   bias (one depth-map texel per unit of tangent of the incidence angle) and
   the result is filtered per part in texture space, which removes the
   depth-map banding on curved surfaces.
6. **Normal map**: shape detail transferred from the high-detail build of the
   same parameters (8 nearest surface samples, inverse-distance weighted,
   converted into the low mesh's tangent frame, then filtered per part in
   texture space) blended with material micro-relief from height fields;
   OpenGL convention.
7. **Gutters**: pixels within 8 px of an island copy their nearest island
   texel (edge padding, so mip levels do not mix neighbouring islands); the
   rest is filled with a mask-weighted push-pull pyramid.

Maps are only written when meaningful (metalness only with metal, emissive
only with emitters, AO only when it varies, normal only with relief).

## Export (exporters/)

- **FBX** (`fbx.py`): binary 7.4 writer and reader, Blender-equivalent node
  structure, embedded PNGs, relative texture paths; verified by re-import in
  Blender 5.2.
- **RBXM / RBXMX** (`rbxm.py`): binary format version 0 with META, INST, PROP,
  PRNT (depth-first post-order) and END chunks, interleaved/zig-zag integers,
  Roblox float encoding, CFrame rotation IDs; XML version 4 whose numbers are
  the same float32 values (9 significant digits). Verified with rbx-dom
  (binary and XML decode to identical trees).
- **GLB** (`interchange.py`): glTF 2.0, metallic-roughness with packed ORM
  texture, normal and emissive textures, V flipped to glTF's convention, and
  the per-vertex tangents the normal map was baked with.
- **OBJ/MTL** with PBR extension keys; **STL** binary.
- **Previews** (`previews.py`): software-rendered with key/fill/hemisphere
  light and a percentage-closer shadow map (normal offset plus slope-scaled
  bias, so curved surfaces show no shadow acne); orthographic
  front/side/back, perspective three-quarter, wireframe with depth-tested
  edges, 24-frame turntable GIF,
  contact sheet, and a self-contained `viewer.html` (three.js from a pinned CDN
  URL, model embedded as base64).
- **Packaging** (`packaging.py`): fixed folder layout, metadata JSON, material
  sheet, Luau import helper, deterministic ZIP that skips symlinks and paths
  outside the package.

## Discord layer (bot/)

- `/3d` command group with 10 subcommands; long work is deferred within 3 s.
- Progress: a message edited at most every 2 s with a text progress bar.
- Persistent result buttons are `DynamicItem`s (custom IDs encode action,
  project and version) and keep working after restarts.
- Modals use Label-wrapped text inputs, selects and the File Upload component.
- Delivery respects `attachment_size_limit`; oversized packages are split into
  their most useful files, or signed expiring links when the optional
  download server is enabled.
- Icons are drawn in code with Pillow and registered as application emojis.

## Security

| Concern | Measure |
|---|---|
| Token leakage | Only from environment/.env; never logged (redacting log filter); not passed to worker processes |
| Malicious uploads | Byte-level validation, Pillow bomb guard, pixel and size limits, renamed files, no extraction or execution |
| Path traversal | `safe_join` for every project path; project IDs validated as 8 hex characters; ZIPs skip symlinks |
| Access control | Owner-only (plus configured admins) on every handler, re-checked on each button press |
| Resource exhaustion | Per-user job limit, global worker slots, per-job timeout, RLIMIT_AS on Linux, capped BLAS threads |
| Database integrity | SQLite WAL, transactions for version numbering, integrity check in `--check` |
| Download links | HMAC-SHA256 signed, expiring, path-contained, opt-in |
