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

## Reconstruction (engine/reconstruction.py, engine/fitting.py)

### Strategy selection
`rank_categories` scores categories from the features; the six best
templates are fitted coarsely. The two best coarse fits plus the best
feature-ranked template (or the third coarse fit) are refined with up to 170
evaluations each, and the lowest refined score wins if it is below the
acceptance score:

| Silhouette | Acceptance | Reason |
|---|---|---|
| lathe-like: mirror symmetry > 0.9 and > 95 % of rows are one run | 0.80 | a surface of revolution reproduces it exactly |
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
- **Inflate**: signed Euclidean distance of the silhouette, half thickness
  `sqrt(d/dmax) * depth/2`; the volume is stored as a continuous field
  (`half_thickness - |z|` inside, distance outside), not as voxels.
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
- **Meshing of fields** (inflate, visual hull, turntable): Gaussian
  smoothing (0.9 voxel for reference views, 0.6 for photo sets, whose thin
  parts are only a few cells wide), always on the full grid, marching cubes
  at level 0, Taubin smoothing; shading
  normals come from the field gradient, so neither the voxel grid nor the
  marching-cubes facets print bands into the baked normal map. After
  decimation the low mesh takes its shading normals from the dense surface.

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
