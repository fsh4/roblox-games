# Image-to-3D Studio

A Discord bot that turns photos into **editable, Roblox-ready 3D models** - FBX
with embedded PBR textures, glTF, a native-part `.rbxm`, separate texture maps,
previews and full metadata - using **classical computer vision and procedural
modelling only**. There is no AI model, no AI API and no pretrained network
anywhere in the pipeline.

```
/3d create image:<photo>   ->   analysis -> reconstruction -> textures -> Roblox optimisation -> package
/3d edit                   ->   new triangle budget / size / materials / textures without re-uploading
```

> Honest scope: one photo shows one side of an object. Visible proportions,
> colours and materials are measured; hidden sides are estimated from symmetry
> and object-type knowledge. Expect a faithful *game asset* of the object, not
> a photogrammetry scan. Accuracy and limits are documented in
> [docs/TEST_REPORT.md](docs/TEST_REPORT.md).

## What you get for every model

```
Bed/
  model.fbx                  binary FBX 7.4, textures embedded, studs, Y up, front -Z
  model.glb                  glTF 2.0 with PBR (metallic-roughness, normal, occlusion)
  model.rbxm / model.rbxmx   native Roblox Part version (opens with Insert from file, no upload)
  textures/                  basecolor, normal (OpenGL), roughness, [metallic], ambient_occlusion, [emissive], [opacity]
  materials/                 materials.json, material_info.md (SurfaceAppearance mapping)
  geometry/                  mesh_statistics.json, geometry_settings.json, mesh_metadata.json
  roblox/                    roblox_import_settings.json, apply_import_settings.luau, collision.obj
  previews/                  front, side, back, three_quarter, wireframe, preview sheet, turntable.gif, viewer.html
  source/                    original image(s), template fit overlay, generation_parameters.json
  README.md
Bed_v1.zip                   everything above in one archive
```

Maps that would carry no information are not written (no metallic map for a
wooden bed, no emissive map without a light source).

## Requirements

| | |
|---|---|
| Python | 3.11 - 3.13 (developed and tested on 3.13.18) |
| OS | Linux, Windows or macOS. Worker memory limits are applied on Linux only. |
| CPU / RAM | 2+ cores, 4 GB RAM minimum; 8 GB recommended for two parallel jobs |
| Disk | about 5-15 MB per model version |
| Discord | a bot application with the `bot` and `applications.commands` scopes |
| Blender | **not required**. Optional, only for the FBX cross-check in tests (`pip install bpy==5.2.3` in a separate venv, or a Blender install) |

## Installation

```bash
git clone <your fork>
cd ImageTo3DStudio
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows
.venv\Scripts\activate
pip install -r requirements.txt
```

## Discord setup

1. Open <https://discord.com/developers/applications> and create an application.
2. **Bot** tab: *Reset Token*, copy it. No privileged intents are needed
   (the bot reads attachments from slash commands and modals, not from messages).
3. **Installation** / **OAuth2** tab: scopes `bot` and `applications.commands`;
   permissions *Send Messages*, *Embed Links*, *Attach Files*, *Read Message History*.
   Open the generated URL and add the bot to your server.
4. `cp .env.example .env` and set `DISCORD_TOKEN`. For fast testing also set
   `DISCORD_GUILD_ID` to your server ID (commands appear instantly; global
   commands can take up to an hour).

## Start

```bash
python main.py --check   # validates config, dependencies and the database
python main.py
```

The first start uploads the code-drawn icon set as application emojis
(`REGISTER_APP_EMOJIS=true`). Data lives in `DATA_DIR` (default `./data`):
`studio.db` (SQLite, WAL), `projects/<id>/source|versions/vNNNN/` and `logs/`.

## Commands

| Command | What it does |
|---|---|
| `/3d create image:[photo] ...` | New model from 1-4 attachments (or open the upload form with up to 6). Options: `category`, `mode`, `views`, `triangles`, `texture`, `quality`, `name`. |
| `/3d edit [project]` | Edit panel: triangle budget, size and scale, textures, shading and bevel, materials and colours, shape parameters, Roblox options (collision, pivot, mesh grouping), lower-poly or higher-detail version, rebuild, re-analyse. |
| `/3d export [project] [format] [version]` | FBX, GLB, OBJ (+MTL+textures), STL, RBXM, RBXMX, textures ZIP or full ZIP. |
| `/3d versions [project]` | Every version with its settings; restore any of them as a new version. |
| `/3d history` / `/3d open` | Your projects / one project card with preview and buttons. |
| `/3d status` | Running and queued jobs, queue position, cancel. |
| `/3d settings` | Your defaults (budget, texture size, quality, public/private results, collision). |
| `/3d delete project` | Delete a project after confirmation. |
| `/3d help` | Short guide. |

Result messages carry buttons (Download ZIP, Files, Edit, Versions, Formats)
that keep working after a bot restart. Only the owner (and `ADMIN_USER_IDS`)
can open or edit a project.

### Photo tips

- Whole object in frame with some background around it; plain backgrounds
  segment best. A PNG with transparency skips background removal entirely.
- About 20-35 degrees above the object for furniture, straight-on for round or
  flat objects.
- Several photos: either label them (`views: front,side,back,top`, orthographic-like
  shots) for a visual hull, or walk around the object at a constant height for
  the photo-set mode. Keep the steps between photos about equal (for example 8
  photos, one every 45 degrees): objects without surface texture are
  calibrated from their silhouettes under that assumption.
- A flat drawing or logo (flat colours, outline) is extruded into a solid
  emblem; holes in it are kept.
- Toys, food, characters and other free-form props become a rounded solid:
  every part is as deep as it is wide in the photo, the photo is wrapped
  around it and mirrored onto the back. A straight-on view works best; change
  the depth afterwards with `/3d edit` > Size.
- Uploading the same photo twice does not help: repeats are ignored, and
  fewer than four unlabelled photos are treated as one photo.
- The unseen back is textured from the materials of the photo (cake layers
  continue, fur stays fur) without copying unique details such as eyes or a
  logo. Holes you can see through (a mug handle) stay holes.
- A busy background (a street, a room) is handled, but the object alone on a
  plain background always gives the best model.

## Importing into Roblox Studio

1. *File > Import 3D*, choose `model.fbx` (or `model.glb`). Keep
   *File Geometry > Scale Unit = Studs* - the files are authored in studs.
2. The textured mesh arrives as MeshPart(s) with its textures (embedded).
3. Select the imported model and paste `roblox/apply_import_settings.luau` into
   the Command Bar to apply the recommended CollisionFidelity, RenderFidelity
   and anchoring.
4. Alternatively *File > Insert from file* with `model.rbxm` for the native-part
   version (no upload, Roblox materials, exact colliders).

Roblox limits respected: 20,000 triangles per mesh (bigger budgets are split
into several MeshParts), single material per mesh (one texture atlas), OpenGL
tangent-space normal maps, 1 stud = 0.28 m for real-world sizes.

## Running the tests

```bash
pytest -q                                   # full suite (several minutes: real jobs run)
pytest -q tests/test_geometry.py            # fast unit tests only
RBXCHECK=/path/to/rbxcheck pytest -q        # also decode RBXM/RBXMX with rbx-dom
BPY_PYTHON=/path/to/bpy-venv/python pytest  # also import every FBX into Blender
```

`tools/quality_report.py` regenerates the numbers in `docs/TEST_REPORT.md`;
`tools/make_test_images.py` re-renders the synthetic test photos with Blender.

## Troubleshooting

| Problem | Fix |
|---|---|
| `DISCORD_TOKEN is not set` | Create `.env` from `.env.example`. |
| `Discord rejected the token` | Reset the token in the Developer Portal; tokens are invalidated when leaked. |
| Commands do not appear | Set `DISCORD_GUILD_ID` for instant registration, or wait for global sync; make sure the bot was invited with `applications.commands`. |
| "Busy" | Each user may have `MAX_JOBS_PER_USER` running/queued jobs. |
| Job "timed out" | Raise `JOB_TIMEOUT_SECONDS`, use quality `draft`, or a smaller texture size. |
| ZIP not attached | The package exceeds the channel's upload limit; use the Files button, `/3d export`, a lower texture resolution, or enable the private download server. |
| Model looks like the wrong object | `/3d edit` > *Re-analyse photo* and pick the object type; or retake the photo on a plain background. |
| Colours wrong | `/3d edit` > *Materials and colours* (hex colour, kind, roughness, metalness). |
| `out of memory` | Lower `MAX_CONCURRENT_JOBS` or raise `WORKER_MEMORY_MB`. |

## Documentation

- [docs/RESEARCH.md](docs/RESEARCH.md) - what was researched and why each technology was chosen or rejected
- [docs/TECHNICAL.md](docs/TECHNICAL.md) - architecture, algorithms, data formats
- [docs/TEST_REPORT.md](docs/TEST_REPORT.md) - measured results, known issues, limitations
- [docs/FEATURE_STATUS.md](docs/FEATURE_STATUS.md) - what is done, tested, partial or unsupported

## Licence

MIT, see [LICENSE](LICENSE).
