# Research and technology selection

Date: 2026-11-08/09. Sources were read from primary documentation where
possible: the Roblox Creator Hub documentation repository
(`github.com/Roblox/creator-docs`), the Discord API documentation repository
(`github.com/discord/discord-api-docs`), the rbx-dom format specification
(`github.com/rojo-rbx/rbx-dom/docs`), and Blender 5.2's own FBX exporter source
(`io_scene_fbx/encode_bin.py`, `export_fbx_bin.py`).

## 1. What one photograph can and cannot give

| Available from one image | Not available |
|---|---|
| Silhouette (outline) of the visible side | Depth of each pixel (requires multiple views, active sensing or a learned prior) |
| Straight edges, their vanishing directions, hence camera tilt and the object's relative proportions (single-view metrology) | Geometry behind the object, inside concavities, under occluders |
| Colours, texture statistics and grain direction of visible surfaces | True scale (no reference length) |
| Symmetry of the outline | Material reflectance separated from lighting (only approximated) |

Consequences for the design:
- Without a learned depth prior, reliable single-image 3D needs a **shape
  model**. Two kinds are used: *parametric templates* (furniture, buildings,
  vehicles - objects that are combinations of boards, cylinders and cushions)
  and *generic shape assumptions* (rotational symmetry, flat extrusion,
  rounded inflation) for everything else.
- Hidden sides are produced by symmetry and the template's structure, and the
  output says so.
- Absolute size comes from the template's real-world prior (a single bed is
  about 2.1 m long) or from the user; 1 stud = 0.28 m (Roblox units page).

## 2. Approaches compared

| Approach | Classical? | Works from one photo | Strengths | Weaknesses | Decision |
|---|---|---|---|---|---|
| Parametric template + analysis-by-synthesis fitting | Yes | Yes | Clean topology, exact triangle budgets, editable parameters, watertight parts, Roblox-native part version for free | Limited to modelled categories | **Used** (Mode C) |
| Surface of revolution from silhouette (Wong, Mendonca, Cipolla 2003) | Yes | Yes | Very accurate for vases, bottles, lamps, barrels | Only rotationally symmetric objects | **Used** |
| Extrusion of the outline | Yes | Yes | Drawings, signs, flat props, weapon sketches | No depth variation | **Used** |
| Silhouette inflation (Teddy-like distance-field inflation) | Yes | Yes | Any blob-like organic object | Back side is a mirror; no concavities | **Used** |
| Visual hull / space carving from labelled views | Yes | No (2+ views) | Robust, no feature matching needed | Cannot recover concavities | **Used** (Mode B) |
| Structure from Motion + MVS (OpenCV SIFT, essential matrix, triangulation) | Yes | No | True geometry when photos overlap and have texture | Fails on plain/featureless objects and casual photos | **Partial**: camera yaw from SIFT + essential matrices, then silhouette carving |
| Shape from shading | Yes | Yes | Fine relief | Needs known lighting/albedo; unstable on real photos | Rejected |
| Monocular depth networks (MiDaS, DPT, Depth Anything), image-to-3D generators (TripoSR, Meshy, etc.) | No - learned | Yes | - | Pretrained neural networks / AI services | **Rejected (requirement)** |
| Segment Anything, rembg/U2-Net background removal | No - learned | - | - | Learned models | **Rejected**; GrabCut + edges + colour models used instead |

## 3. Dependencies evaluated

| Library | Licence | Maintenance | Used for | Notes |
|---|---|---|---|---|
| discord.py 2.7.1 | MIT | Active (2.7 adds modal file uploads, Components V2) | Bot, slash commands, modals with FileUpload/Select, DynamicItem persistent buttons, application emojis | Chosen over Pycord/nextcord: largest user base, supports every component needed |
| NumPy / SciPy | BSD-3 | Active | All geometry, KD-trees, sparse graphs, special functions | |
| OpenCV (opencv-python-headless 4.14) | Apache-2.0 | Active | GrabCut, Canny, LSD lines, k-means, SIFT, essential matrix, homographies | OpenCV 5.0 exists; 4.x kept for API stability. SIFT is patent-free since 2020 |
| Pillow | MIT-CMU | Active | Safe image decoding (decompression-bomb guard), PNG/GIF output | |
| scikit-image | BSD-3 | Active | Marching cubes for carved/inflated volumes | |
| trimesh | MIT | Active | Convex hulls for collision proxies; independent re-loading of OBJ/STL/GLB in tests | Not used for core geometry |
| fast-simplification | MIT | Active | Quadric edge-collapse decimation (C++), the same backend trimesh uses | Pure-Python QEM was too slow for 100k-triangle carved meshes |
| mapbox-earcut | ISC | Active | Polygon (with holes) triangulation for extrusions | |
| Open3D | MIT | Active | - | Not used: 400 MB wheels; Poisson reconstruction not needed for silhouette-based methods |
| PyMeshLab | GPL | Active | - | Not used: GPL and large; QEM covered by fast-simplification |
| xatlas | MIT | Active | - | Not needed: procedural parts carry their own UVs; carved meshes use cube-projection charts |
| Blender (bpy 5.2.3 from PyPI) | GPL | Active | **Optional** external verification: our FBX files are imported in Blender in the test suite | Not a runtime dependency; the bot writes FBX itself |
| rbx-dom (Rust crates rbx_binary, rbx_xml) | MIT | Active (used by Rojo and Lune) | Test-only validator that decodes our .rbxm/.rbxmx | Small Rust tool built from crates.io |

## 4. Roblox import facts (Creator Hub docs, read 2026-11-08)

- Individual meshes may not exceed **20,000 triangles**; geometry must be
  watertight with volume (`art/modeling/specifications.md`).
- One material per mesh object; PBR via SurfaceAppearance: ColorMap,
  NormalMap (**OpenGL tangent space only**), RoughnessMap, MetalnessMap,
  EmissiveMaskContent; there is **no ambient-occlusion slot**, so AO is baked
  mildly into the colour map and also delivered separately.
- Textures up to 4096 px are accepted; guidance is 256 px for ~5-stud objects,
  512 for ~10, 1024 for ~20 (`texture-specifications.md`).
- Blender export settings Roblox documents: FBX, Path Mode *Copy* + *Embed
  Textures*, Apply Scalings *FBX Unit Scale*, Forward *Z*, Up *Y*, no leaf
  bones. The importer's Scale Unit defaults to *Studs*.
- Importer options: Set Pivot to Scene Origin (default on), Anchored, Merge
  Meshes, World Forward/Up.
- CollisionFidelity: Box, Hull, Default, PreciseConvexDecomposition, Tunable;
  not settable by running game scripts, so the package includes a Command Bar
  helper to apply it in Studio.
- A MeshPart can only reference meshes hosted by Roblox, so a `.rbxm` cannot
  embed our triangle mesh. The `.rbxm/.rbxmx` therefore contain a native-part
  build (Block/Cylinder/Ball parts with Roblox materials); the textured mesh is
  delivered as FBX/GLB for the 3D Importer.

## 5. FBX format decisions

Blender's exporter was run with Roblox's documented settings and the resulting
file was decoded node by node. The studio's writer reproduces that structure:
binary FBX 7400, GlobalSettings (UpAxis +Y, FrontAxis -Z, CoordAxis -X,
UnitScaleFactor 100), geometry in Blender object space under a model node
rotated (90, 0, 180), a Phong material whose texture connections use
Blender's slots (DiffuseColor, NormalMap, ShininessExponent for roughness,
ReflectionFactor for metalness, EmissiveColor) and PNGs embedded in
`Video.Content`. Matching Blender byte-structure-for-structure means any
importer that handles Blender's FBX (Roblox documents that path) handles ours.

## 6. Discord facts (discord-api-docs, read 2026-11-08)

- Interactions must be acknowledged within **3 seconds**; tokens are valid for
  **15 minutes** (deferred responses + follow-ups are used; long jobs fall back
  to channel messages or DMs).
- Default upload limit is **20 MiB per attachment** since 2026-09-03 (10 MiB
  before), checked per attachment; the interaction payload carries
  `attachment_size_limit` (boosted servers are higher). The bot reads that value.
- Modals support File Upload components (0-10 files) and selects inside labels.
- Applications can own up to 2000 emojis - used for the code-drawn icon set.

## 7. What is realistically achievable without AI

| Input | Expected result |
|---|---|
| Furniture, simple buildings, simple vehicles, one clear photo | Correct object type, proportions within roughly 10-20 %, measured colours and wood grain, clean game-ready topology. Hidden sides follow the template. |
| Round objects (vase, lamp, bottle) | Close match of the outline; texture projected from the photo. |
| Flat objects and drawings | Faithful outline; front and back from the image. |
| Organic objects (toys, fruit, characters) from one photo | Recognisable rounded volume; back side mirrored; no limbs separated in depth. |
| Several labelled views | Visual hull: correct outer shape, no concavities. |
| Casual photo sets | Works when photos overlap and have texture; otherwise falls back to single-image mode with an explanation. |
| "1:1 exact detail" of arbitrary objects from one photo | **Not achievable** with classical methods (or any method without priors); the system says what was estimated. |
