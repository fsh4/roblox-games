# Feature status

Every feature carries exactly one label. "Tested" means an automated test in
`tests/` or a measured run in [TEST_REPORT.md](TEST_REPORT.md) exercised it;
the evidence column says which.

| Label | Meaning |
|---|---|
| **Done + tested** | Completed and verified by tests or measured runs |
| **Done, not fully tested** | Implemented, but some part could not be verified here |
| **Partial** | Works for part of the intended scope |
| **Not implemented** | Not built |
| **Unsupported** | Not possible with the chosen technology (no AI, one photo, Roblox or Discord limits) |

## Input and image analysis

| Feature | Status | Evidence / notes |
|---|---|---|
| Upload validation (magic bytes, size, pixel count, decompression bombs, renamed files) | Done + tested | `test_security.py` |
| EXIF orientation and EXIF focal length | Done, not fully tested | Code path used on every photo; no test photo carries EXIF |
| Large images downscaled to 1024 px for analysis | Done + tested | `test_large_image_is_downscaled` |
| Background removal: plain studio, room scene, cluttered real photo, PNG alpha | Done + tested | bed photo, chair room render, plush alpha PNG, turntable JPEGs |
| Low-contrast fallback (centred graph cut) | Done, not fully tested | No test image triggers it |
| Feature extraction (symmetry, lines, palette, holes, drawing score, row convexity) | Done + tested | Drives the automatic selection test |

## Reconstruction

| Feature | Status | Evidence / notes |
|---|---|---|
| Automatic strategy and category selection | Done + tested | 6 of 6 single-image test inputs classified correctly (bed, chair, table, vase, plush, emblem); accuracy on arbitrary photos is unknown |
| Template fitting by analysis-by-synthesis: bed, chair, table | Done + tested | Real bed photo and two renders; fit scores in the test report |
| Templates sofa, cabinet, shelf, lamp, house, car, box | Done, not fully tested | Geometry tested (closed solids, resizing); never fitted to a photo of that object type |
| Repeated element counting (slats, rows, shelves, cushions) | Done, not fully tested | Runs on every template fit; counts not asserted by a test |
| Materials and colours measured from the photo | Done + tested | Bed frame detected as wood (test); colours checked against the photos |
| Photo exemplar textures (wood, stone, leather) | Done + tested | `test_exemplar_synthesis_keeps_colour_statistics`; used on the bed |
| Surface of revolution (vases, bottles, lamps) | Done + tested | Vase, chosen automatically |
| Extrusion of flat artwork with holes | Done + tested | Emblem, chosen automatically, centre hole kept |
| Free-form objects: medial-ball inflation, flat base, wrap projection, photo relief | Done + tested | Egg photo and plush, chosen automatically; `tests/test_sculpt.py` (disc gives a sphere within 12 % volume, thin parts stay thin) |
| Synthesised back (camera elevation, lighting-free quilting, no copied details) | Done + tested | `tests/test_surround.py`: elevation of the egg photo found (15-40 degrees), back differs from the mirrored front, azimuth continuous on the body; egg, plush, vase, donut and mug samples checked from four sides |
| See-through openings kept (handles, rings) | Done + tested | `test_see_through_hole_is_kept`: donut mesh has exactly one hole |
| Busy backgrounds and letterboxed pictures | Done, not fully tested | `test_letterbox_bars_are_removed`; centred graph cut checked by eye on five street and nature photos (not in the repository) |
| Template fits that spill onto the background are rejected | Done, not fully tested | Two-car photo no longer becomes a sofa; no automated test |
| Templates only for outlines with straight edges | Done + tested | `test_decorated_egg_is_not_furniture`: no template is tried on the egg photo |
| Repeated or too few unlabelled photos fall back to one photo | Done + tested | `test_repeated_photo_does_not_start_a_photo_set` |
| Visual hull from labelled views (front/side/back/left/top) | Done + tested | Mug from 4 views |
| Photo set walked around the object (unlabelled) | Done + tested | 8 chair photos: steps recovered as 45.0 degrees (true 45), elevation 15 (true 14); legs and the three back slats kept |
| Photo set: SIFT and essential-matrix path | Done, not fully tested | Implemented; the test set has too little texture, so the silhouette-coherence fallback is what was measured |
| Concave regions (inside of a cup, under a chair seat) | Unsupported | No silhouette shows them; one photo cannot measure them |
| Exact 1:1 replica of hidden sides from one photo | Unsupported | Hidden sides are estimated from symmetry and object knowledge |
| Depth estimation by neural networks | Unsupported | Excluded by the no-AI rule |

## Geometry, optimisation, Roblox rules

| Feature | Status | Evidence / notes |
|---|---|---|
| Triangle budget (LOD search, then quadric decimation) | Done + tested | Budgets 100 to 5,000 in tests; 8 measured runs |
| Unreachable budgets reported with the minimum | Done + tested | `test_unreachable_budget_is_reported` |
| Quality-impact metric of a reduction | Done + tested | Reported for every measured run |
| Split above 20,000 triangles per MeshPart | Done + tested | `test_split_for_roblox_respects_meshpart_limit` |
| Watertight, outward-wound meshes, validation report | Done + tested | Every measured run |
| Pivot options (bottom centre, centre, back, origin) | Done, not fully tested | Centre and default tested; back and origin not asserted |
| Collision proxy and CollisionFidelity recommendation | Done, not fully tested | Written for every run; behaviour inside Roblox not checked |
| Smooth / flat shading, bevel and rounding | Done + tested | Normal tests in `test_geometry.py`; edit test |
| Size in studs, scale, per-axis dimensions | Done + tested | `test_template_resizing_hits_target_dimensions`, edit test |

## Textures

| Feature | Status | Evidence / notes |
|---|---|---|
| Single atlas, physical texel density, non-overlapping islands | Done + tested | `test_atlas_islands_do_not_overlap` |
| Procedural materials (wood, fabric, metal, leather, stone, plastics) | Done + tested | `test_wood_material_varies_but_keeps_mean`; visual checks |
| Photo projection with visibility and mirrored hidden sides | Done + tested | Generic strategies in every measured run |
| Base colour, normal (OpenGL), roughness, metallic, AO, emissive, opacity | Done + tested | `test_bake_produces_meaningful_maps`, `test_metallic_map_skipped_without_metal` |
| Maps omitted when they carry no information | Done + tested | Same tests |
| Seamless gutters (push-pull) | Done + tested | `test_push_pull_fills_every_pixel` |

## Export and package

| Feature | Status | Evidence / notes |
|---|---|---|
| FBX 7.4 binary, embedded textures, studs, Blender-compatible | Done + tested | Round trip, import in Blender 5.2 with texture links (every measured run) |
| Import of the FBX into Roblox Studio | Done, not fully tested | Roblox Studio does not run on the Linux build machine; settings follow Roblox's documented importer |
| RBXM / RBXMX native-part model | Done + tested | Decoded with rbx-dom; binary and XML give identical trees |
| Opening the RBXM in Roblox Studio | Done, not fully tested | Same reason as above |
| MeshPart with the textured mesh inside the RBXM | Unsupported | A MeshPart needs a mesh asset ID that only exists after uploading to Roblox |
| Automatic upload to Roblox (Open Cloud) | Not implemented | Needs the owner's Open Cloud API key; the FBX is imported by hand |
| GLB, OBJ + MTL, STL | Done + tested | `test_glb_is_valid_gltf`, `test_obj_and_stl_roundtrip` |
| Previews: orthographic views, three-quarter, wireframe, contact sheet, turntable GIF | Done + tested | Generated and inspected in every measured run |
| Interactive `viewer.html` (three.js) | Done + tested | Opened in headless Chromium: the textured model loads and renders with no console errors (needs internet for the three.js library) |
| Metadata files, Luau import helper, README per package | Done + tested | `test_package_is_complete_and_within_budget` |
| Deterministic ZIP without symlinks or outside files | Done + tested | `test_zip_contains_only_package_files` |
| Re-export from a stored version in another format | Done + tested | `test_export_conversion_from_package`, bot flow test |

## Discord bot, storage and jobs

| Feature | Status | Evidence / notes |
|---|---|---|
| `/3d` command group (create, edit, export, history, versions, open, status, settings, delete, help) | Done + tested | Definitions validated offline (`test_slash_commands_are_valid`) |
| Create, progress, result, edit, export and access control flow | Done + tested | `test_create_edit_export_flow` with stand-in Discord objects |
| Behaviour against the real Discord API (gateway, attachments, File Upload modal, emoji upload) | Done, not fully tested | No bot token in the build environment; run `python main.py --check` and a live smoke test after setup |
| Persistent buttons after a restart | Done + tested | Custom-ID parsing and `DynamicItem` registration tested; live restart not tested |
| Upload-limit aware delivery and signed download links | Done + tested | `test_signed_links`, `test_download_server_serves_only_signed_files` |
| Edit panel: budget, size, textures, shading, materials, shape, Roblox options, lower/higher detail, rebuild, re-analyse | Done, not fully tested | Engine side tested for budget, size, materials, pivot; each Discord modal checked for limits, not clicked live |
| Versions and restore | Done + tested | `test_projects_versions_and_ownership`, edit test |
| Job queue: concurrency, isolation of failures, timeout, cancel, restart recovery | Done + tested | `test_storage_jobs.py` |
| Per-user limits, owner-only access, token never logged or passed to workers | Done + tested | Bot flow test, logging filter, queue environment |
| 3D view inside Discord | Unsupported | Discord cannot render 3D; the bot sends images, a GIF and `viewer.html` |
