# FCTX MATTER STUDIO — architecture

> Move your bare hands in front of a camera and the matter on screen answers.
> Pinch a sheet of cloth and it creases where your fingers are. Push a soft
> body and it dents. Slide the hardness dial while you are still holding it and
> the same shape stops behaving like jelly and starts behaving like rubber —
> not just in colour, but in how it wobbles when you let go and how it hits the
> floor.

This document is the contract between the four subsystems. It is written so
that each can be built and tested without the others existing yet.

---

## 1. The idea in one paragraph

Everything on screen is one particle system. Cloth, soft bodies and granular
matter are not three engines; they are one XPBD solver running different
constraint sets over the same buffers. That is what makes the showpiece
possible: "hardness" is a single scalar that rewrites the *compliance* of every
constraint in the scene, live, mid-grab, without rebuilding anything and
without touching the numerics. Compliance is physical (it is 1/stiffness, and
XPBD folds it into the solve as `α̃ = α/Δt²`), so the feel of the material does
not change when the solver takes more or fewer substeps.

---

## 2. Data flow

```
 camera ──► hands/ ─────────────► HandPose[]  ──┐
 (or synthetic / replay)                        │
                                                ▼
 SceneConfig ──► bodies/ ──► BodyData[] ──► solver/ ──► GPU particle state
                                                │              │
                                                │              ▼
 hardness dial ──► core/material ──► Material ──┘         render/ ──► screen
```

* `core/types.py` and `core/material.py` are imported by everyone and import
  nobody. They must stay free of `warp`, `moderngl`, `glfw` and `mediapipe` so
  the CPU-only tests run on any machine.
* `bodies/` is pure numpy. It never imports `warp`.
* `solver/` is the only module that writes to GPU physics buffers.
* `render/` never mutates physics state; it reads `SolverState` buffers and
  `HandPose` objects.

---

## 3. Coordinate frames

| frame | units | description |
|---|---|---|
| image | normalised | MediaPipe landmarks. x,y ∈ [0,1], origin top-left. z relative, negative = nearer the camera. |
| hand-local | metres | MediaPipe `world_landmarks`. Origin at the hand's geometric centre. Carries shape, not position. |
| world | metres | Simulation space. +X right, +Y up, +Z toward the viewer. Ground plane at y = `SolverConfig.ground_y`. |

The stage — the volume a tracked hand can reach — comes from `TrackingConfig`.
By default it is 0.84 m wide and 0.52 m tall about (0, 0.30), i.e. y from 0.04
to 0.56, with depth clamped to `z_range = (−0.32, 0.48)` about the reference
plane `stage_center_z = 0.10`. A body is 0.26–0.66 m across (the torus and the
soft sphere at 0.259, the banner at 0.660). Those numbers are chosen so that a
real hand at arm's length from a webcam maps onto something the size of a
folded towel — small enough to grab with two fingers, big enough to see fold.

**A preset may move the stage, and one does.** The stage has to contain the
matter, not the other way round: `grain` settles into a 0.13 m basin on the
floor, so it lowers the stage to 0.40 m tall about y = 0.19 (reach 0.01 to
0.39 m). Leaving the default stage there would put the interaction volume at
chest height above a pile on the ground, and the hand could only ever reach
the very top of it. Anything that rebuilds a scene at runtime therefore has to
carry `TrackingConfig` across with `SceneConfig`, not just `SceneConfig`.

---

## 4. Module contracts

### 4.1 `fctx.bodies` — pure numpy, no GPU

```python
def build_scene(scene: SceneConfig) -> list[BodyData]
def build_cloth(scene: SceneConfig, mp: MaterialParams) -> BodyData
def build_soft_body(scene: SceneConfig, mp: MaterialParams) -> BodyData
def build_granular(scene: SceneConfig, mp: MaterialParams) -> BodyData
def greedy_color(indices: np.ndarray, num_particles: int) -> np.ndarray
```

`BodyData` is defined in `core/types.py`; `BodyData.validate()` must pass for
everything this module returns.

**Graph colouring is load-bearing.** The solver runs Gauss–Seidel constraint
projection in parallel. Two constraints that share a particle must never run in
the same kernel launch, or they race and the result is both wrong and
non-deterministic. `greedy_color` partitions constraints so that within a
colour no particle appears twice. Constraints in a `BodyData` do **not** have to
be sorted by colour — `SolverState` sorts them during assembly — but the colour
array must be a valid partition. `tests/test_coloring.py` proves it.

**Cloth** is a regular grid of `cloth_resolution²` particles:
* structural distance constraints along both axes (`ConstraintKind.STRETCH`),
* diagonal distance constraints (`ConstraintKind.SHEAR`),
* dihedral bending constraints across every interior edge, with the rest angle
  measured from the flat initial configuration (so a flat sheet starts at rest).

Pinning is by `cloth_pinned`: `none`, `top_corners`, `two_points`,
`top_edge`, `corners`. Pinned particles get `inv_mass = 0` and `FLAG_PINNED`.

`two_points` is the default, and the difference is not cosmetic. The top edge
of a sheet is inextensible, so pins exactly one width apart hold it dead
straight and the cloth hangs as a smooth shell with no folds at all — it reads
as a curved board. Bringing the pins in to 22% and 78% leaves slack, the top
edge buckles, and the fabric breaks into the diagonal folds people recognise.
Measured on the shipped sheet at hardness 0.3, settled for 6 s: out-of-plane
depth goes from 0.086 m (`top_corners`) to 0.124 m, and the out-of-plane
standard deviation — which is what reads as folding rather than as one bulge —
from 0.010 m to 0.026 m, a factor of 2.7. `top_edge`, pinned all the way
across, is 0.021 m and 0.002 m: flat.

**Soft bodies** are tetrahedralised. Voxelise the implicit shape (sphere, box,
torus) on a lattice of `soft_resolution` cells along the longest axis, keep the
cells whose centre is inside, then split each kept cell into 5 tetrahedra using
the alternating (parity-flipped) 5-tet split so neighbouring cells share faces.
For each tet store the inverse rest shape matrix `Dm⁻¹` where

```
Dm = [x1-x0, x2-x0, x3-x0]   (3×3, columns)
V   = det(Dm) / 6
```

Tets must be wound so `V > 0`; flip two vertices if not. Also emit distance
constraints along tet edges (`STRETCH`) — Neo-Hookean alone is soft against
element inversion at low resolution and the distance constraints cheaply stop
it. The render surface is the set of triangular faces referenced by exactly one
tet, wound outward.

**The drawn surface is not the physics lattice.** A voxelised body cannot be
both smooth and stable with one set of points. Pulling the skin all the way
onto the isosurface is what makes it look like a sphere; the boundary
tetrahedra -- the ones with all four vertices on the skin and nowhere to give
-- are what invert when it is then squeezed. Asking one set of points to do
both jobs means choosing which to sacrifice, and even at `soft_resolution=21`
the fitted lattice still sat 6.0 mm from a true radius, which reads as
Minecraft.

So the two are separated. The lattice takes only a gentle fit, enough to keep
the collision surface near the visible one, and stays well conditioned. Each
surface particle additionally carries `skin_tet` and `skin_bary`, shaped
`(P, K)` and `(P, K, 4)` with `K = 8`: up to eight of the tetrahedra incident
on it, ranked by how interior its *ideal* (fully projected) position is to
each, and the barycentric weights of that position in each element's rest
pose, pre-divided by the count. `kernels.apply_skin` sums them every frame
into `x_skin`, and both the shading normals and the renderer read that rather
than `x`.

Each map is affine per element, so the skin follows every stretch, shear,
twist and dent exactly; a translation, rotation or uniform scale carries it
with no drift at all, which `tests/test_bodies.py` asserts to 1e-9. Blending
over several elements is linear blend skinning: still exact at rest, since
each element's map reproduces the ideal point on its own and so does their
mean, but continuous in the deformation gradient across element boundaries.
Bound to one element the skin showed every boundary as a crease on a
squeezed body -- an orange-peel texture at about 2% of the radius -- and the
blend removes it. Weights outside [0, 1] are normal and deliberate: a skin
vertex up to a cell from its lattice vertex often sits outside every element
touching it, and extrapolating the same affine map is the right answer; the
most-interior ranking just drops the longest lever arms first. The binding
must be computed *after* the winding fix: flipping a tetrahedron swaps two of
its vertices, and weights bound to the old order name the wrong corners,
which looks like a torn mesh rather than a bad index.

Measured on the shipped sphere: the radial spread of the drawn surface falls
from 6.0 mm to 0.12 mm, a factor of 50, for one kernel over the particles and
no change to the physics at all.

**Granular** is a jittered close-packing of `grain_count` spheres in a box,
with no constraints at all — the behaviour comes entirely from
particle-particle contact and friction. `FLAG_SELF_COLLIDE` on every particle.
The initial packing must clear the solver's contact threshold, not merely
avoid geometric overlap: the solver separates particles at
`(rᵢ+rⱼ)·(1 + collision_margin)`, so the packing slack is derived from the
margin rather than hand-tuned.

The `grain` preset stands the scene in a cylindrical basin
(`SolverConfig.basin_radius` / `basin_height`; both are 0, i.e. no basin, in
every other preset). Position-based friction is proportional to penetration
depth, which tends to zero at rest, so an unconfined pile never actually stops
creeping: measured on the shipped 24 000 grains with the basin removed, the
heap is at 8.2° after 3 s and still flattening at 4.8° after 12 s, against a
real granular material's 30-odd, and it spreads until it slides off the stage.
The basin is an honest container rather than a pretence that the friction
model is better than it is, and because the wall stops at `basin_height` a
hand can still lift a fistful straight out.

### 4.2 `fctx.solver` — the only writer of GPU state

```python
class SolverState:
    def __init__(self, bodies: list[BodyData], cfg: AppConfig, device: str)
    # device arrays, described in §5
    def upload_materials(self, materials: list[Material]) -> None
    def snapshot(self) -> dict[str, np.ndarray]     # host copy, for tests
    def reset(self) -> None

class XPBDSolver:
    def __init__(self, state: SolverState, cfg: AppConfig)
    def set_hands(self, poses: list[HandPose], dt: float) -> None
    def begin_grab(self, hand_slot: int, pose: HandPose) -> int   # count grabbed
    def end_grab(self, hand_slot: int, pose: HandPose) -> None
    def update_grabs(self, poses: list[HandPose], now: float) -> None
    def step(self, dt: float) -> None
    def compute_normals(self) -> None
    def reset(self) -> None
    def stats(self) -> dict[str, int]
```

`begin_grab` / `end_grab` are the primitives; `update_grabs` drives them from
a pose list using the hysteresis in `GrabConfig`, for callers that do not want
their own state machine. `fctx.interaction.GripManager` is the one the
application uses, and it calls the primitives directly.

A hand's colliders fade in over `XPBDSolver.COLLIDER_FADE_IN` seconds whenever
a slot starts being tracked. A hand appears wherever the person's hand happens
to be, and in the granular scene that is inside the pile: at full radius on the
first frame, twenty-one capsules materialise inside twenty-four thousand grains
and throw them five metres.

Per `step(dt)`, with `h = dt / substeps`, for each substep:

1. `integrate` — save `x_prev`, apply gravity/wind/drag, `x += v·h`.
2. distance constraints, one launch per colour.
3. bending constraints, one launch per colour.
4. tetrahedron constraints (deviatoric then hydrostatic), one launch per colour.
5. grab attachment constraints.
6. hand capsule collision, ground collision, particle–particle collision.
7. `finalize` — `v = (x - x_prev)/h`, damping, velocity clamp, friction.

XPBD multipliers (`*_lambda`) reset to zero at the start of each **substep**,
not each frame.

The whole substep loop is captured into a CUDA graph when
`SolverConfig.use_cuda_graph` is set. That imposes two rules on the kernels:

* **No host-visible scalars may change between frames.** `dt` is fixed by the
  fixed-timestep clock, so it may be a kernel argument. Anything driven by the
  hardness dial, the hands, or a grab must live in a device array.
* **Launch dimensions are fixed.** Dynamic counts (active capsules, grabbed
  particles) are padded to capacity; the kernel reads a device-side count and
  returns early for inactive lanes.

### 4.3 `fctx.hands` — tracking

```python
class HandSource(abc.ABC):
    def start(self) -> None
    def poll(self) -> TrackerFrame | None    # non-blocking, newest frame
    def close(self) -> None
    @property
    def describe(self) -> str

class CameraSource(HandSource)      # OpenCV capture thread + MediaPipe LIVE_STREAM
class VideoSource(HandSource)       # same, from a file, paced to real time
class SyntheticSource(HandSource)   # procedural hand driven by mouse + keys
class ReplaySource(HandSource)      # .fhr recording playback

class HandTracker:
    def __init__(self, cfg: TrackingConfig, grab: GrabConfig | None = None)
    def update(self, frame: TrackerFrame | None, now: float) -> list[HandPose]
```

`grab` supplies the pinch thresholds and hold time the tracker uses to set
`HandPose.pinching` and to classify the gesture. The tracker holds no matter
of its own; `fctx.interaction.GripManager` is what turns a pinching pose into
a grab.

`HandTracker` owns everything between a raw `TrackerFrame` and a usable
`HandPose`: One-Euro filtering per landmark, image→world projection, velocity
estimation, pinch/curl/gesture classification, stable track ids, and coasting
through dropped frames.

**Smoothing is not cosmetic.** A capsule collider that jitters by 3 mm at 60 Hz
is a 0.18 m/s velocity impulse injected into the cloth. Unfiltered landmarks
make the simulation vibrate, which reads as instability in the physics even
though the physics is fine. The One-Euro filter (Casiez et al., 2012) is the
right tool: it is aggressive when the hand is still and nearly transparent when
the hand is moving, which is exactly the tradeoff this needs.

`CameraSource` must never block the render loop. OpenCV capture runs on its own
thread; MediaPipe runs in `LIVE_STREAM` mode with a result callback; `poll()`
returns the newest completed result or `None`.

### 4.4 `fctx.render` — reads, never writes

```python
class Window:            # glfw + moderngl context, input events
class OrbitCamera:       # view/projection matrices
class Renderer:
    def __init__(self, ctx: Window | moderngl.Context, state: Any,
                 bodies: Sequence[BodyData], cfg: AppConfig)
    def draw(self, camera, poses: list[HandPose], materials: list[Material],
             preview: np.ndarray | None, stats: FrameStats, hud_lines: list[str],
             *, hardness: float | None = None, notifications: Sequence[Any] = (),
             show_hud: bool | None = None, paused: bool = False) -> None
```

`state` is typed `Any` on purpose: the renderer needs `x` and `normal` and
nothing else, so a numpy stand-in renders, which is what lets the graphics be
tested without the solver. The keyword arguments of `draw` are the parts of
the interface the application owns rather than the material: `hardness` is
where the *dial* is (the material lags it), and `notifications` / `paused`
still draw when `show_hud` is false, because hiding the HUD for a clean
recording is exactly when the user still needs to see that the preset changed.

Pipeline per frame:

1. **Shadow pass** — depth-only render of all matter into a 2048² depth texture
   from the key light's point of view, front faces culled so a closed body
   stores its far surface.
2. **Depth prepass** — full resolution, non-multisampled, matter plus the floor
   and the hands. SSAO has to sample a depth texture and this is a forward
   renderer with no G-buffer; the contact where a body meets the ground is the
   most useful thing SSAO contributes and it cannot see a contact whose other
   half is missing from the depth buffer.
3. **SSAO** — half-resolution from that depth texture, blurred. It runs
   *before* the main pass and is sampled by it in screen space, so the
   occlusion lands on the ambient term only.
4. **Main pass** — HDR `f16` colour + depth, MSAA, into an offscreen FBO.
   Cook–Torrance GGX, one key light, one rim light, a hemisphere ambient, PCF
   shadows, and a wrapped-diffuse term scaled by `Material.translucency` so soft
   matter glows from within and hard matter does not.
5. **Resolve** — the multisample resolve is a shader pass rather than a blit,
   so the samples can be weighted by `1/(1 + luma)` (Karis). Averaging HDR
   samples linearly lets one very bright subsample dominate a partly covered
   edge pixel, which leaves the MSAA edge jagged exactly where the contrast is
   highest: the specular highlights on the matter.
6. **Bloom** — progressive downsample/upsample chain (the Call of Duty /
   Jimenez approach), not a fixed two-pass blur; it is what stops the highlights
   from looking like they have a halo stuck on.
7. **Composite** — ACES filmic tonemap, exposure, vignette, chromatic
   aberration, dither to kill banding.
8. **Overlay** — webcam picture-in-picture with the landmark skeleton drawn on
   it, the HUD, and the hardness dial.

**The shadow bias is per receiver kind, not per material.** `shadow.glsl`
offsets the lookup along the normal and the light by a slope-scaled multiple
of the world texel, and how much it may offset depends on what the shadow map
holds for that surface: a closed body drawn back-face only (`SHADOW_CLOSED`)
stores its *far* side, so it can take a far larger offset than a sheet
(`SHADOW_SHEET`), whose own near surface is what it has to avoid. Everything
else — the floor, which never writes to the map at all, and the grain
impostors, which are drawn with culling off and so store their near side —
is `SHADOW_NEAR` and takes the sheet's bias. A grain is 4.8 mm in radius;
given the closed-body multiplier its lookup lands whole grains away, and
measured against a shader tree that does exactly that, 2.2% of the settled
pile's pixels lose more than 20 levels of contact shadow and it stops reading
as touching spheres. `tests/test_render.py` holds that A/B.

Positions and normals reach the GPU through **one** global VBO pair covering
every particle in the scene; each body draws with its own index buffer holding
global particle indices. `wp.RegisteredGLBuffer` maps them so the solver writes
straight into GL memory with no round trip through the host.

---

### 4.5 `fctx.settings` — the configuration file

```python
def load_config(path, base: AppConfig | None = None,
                ignore_preset: bool = False) -> AppConfig
def apply_toml(data: dict, base=None, where="config", ignore_preset=False) -> AppConfig
def dump_config(cfg: AppConfig, preset_name: str | None = None) -> str
class ConfigError(ValueError)
```

A TOML document whose tables are the dataclasses hanging off `AppConfig`
(`scene`, `solver`, `grab`, `tracking`, `render`, `camera`) and whose keys are
their fields. `preset = "..."` at the top names the starting point. The CLI
layers preset → file → flags, and `--preset` on the command line wins over the
file's `preset` (`ignore_preset`) while the file's sections still apply.

**Unknown keys are errors, not warnings.** A venue file exists so that a
camera index or a hand span survives the next morning's restart; a typo that
was silently skipped would defeat that at exactly the moment nobody is
looking. The error names the key and, for a section field, the nearest real
one. Values are coerced to the field's declared type — lists to tuples of
the declared length, strings to `MatterKind`, strings to `Path` — and refused
otherwise. `dump_config` writes the effective configuration back out, and
`tests/test_settings.py` proves the round trip is exact.

## 5. `SolverState` device arrays

`P` = total particles, `D` = distance constraints, `B` = bending constraints,
`T` = tetrahedra, `NB` = bodies, `H` = `max_hands`, `C` = `H * 21` capsule
capacity, `G` = `H * GrabConfig.max_particles` grab capacity.

| name | dtype | size | notes |
|---|---|---|---|
| `x` | `vec3` | P | current position |
| `x_prev` | `vec3` | P | position at the start of the substep |
| `x_rest` | `vec3` | P | initial position, for reset |
| `v` | `vec3` | P | velocity |
| `w` | `float` | P | inverse mass, **current** (a grab *raises* it: `w = w_rest / mass_scale` with `mass_scale < 1`, so held matter follows the hand more readily) |
| `w_rest` | `float` | P | inverse mass with no grab applied |
| `flags` | `uint32` | P | `FLAG_*` bits |
| `body` | `int32` | P | which body owns this particle |
| `radius` | `float` | P | render radius, and the length scale every correction clamp is measured in |
| `collide_radius` | `float` | P | contact radius: `radius` capped at a quarter of the shortest bonded edge, so a bonded neighbour is not a permanent contact |
| `x_skin` | `vec3` | P | the *drawn* position; aliases `x` when no body has a render skin |
| `skin_tet` | `int32` | P | element this particle is drawn inside, -1 for "draw it where it is" |
| `skin_bary` | `vec4` | P | barycentric weights in that element's rest pose |
| `normal` | `vec3` | P | smooth shading normal, computed from `x_skin` |
| `contact_n` | `vec3` | P | normal of the collider contact that demands the most separation this substep; `finalize` reads it |
| `contact_vn` | `float` | P | the normal velocity that contact asks for, restitution included |
| `contact_dx` | `vec3` | P | the averaged particle–particle correction of §6.5, written by `solve_particle_contacts` and applied in a second pass |
| `dist_idx` | `vec2i` | D | sorted by colour |
| `dist_rest` | `float` | D | |
| `dist_kind` | `int32` | D | `ConstraintKind` |
| `dist_body` | `int32` | D | for the per-body compliance lookup |
| `dist_lambda` | `float` | D | |
| `bend_idx` | `vec4i` | B | `(e0, e1, w0, w1)`, sorted by colour |
| `bend_rest` | `float` | B | rest dihedral angle |
| `bend_body` | `int32` | B | |
| `bend_lambda` | `float` | B | |
| `tet_idx` | `vec4i` | T | sorted by colour |
| `tet_dm_inv` | `mat33` | T | |
| `tet_volume` | `float` | T | rest volume |
| `tet_body` | `int32` | T | |
| `tet_lambda_d` | `float` | T | deviatoric multiplier |
| `tet_lambda_h` | `float` | T | hydrostatic multiplier |
| `mat_stretch` | `float` | NB | compliance, m/N |
| `mat_shear` | `float` | NB | compliance, m/N |
| `mat_bend` | `float` | NB | compliance, rad/(N·m) |
| `mat_dev` | `float` | NB | deviatoric compliance `1/mu`; the kernel divides by each element's rest volume |
| `mat_hyd` | `float` | NB | hydrostatic compliance `1/lambda`; likewise per element |
| `mat_grab` | `float` | NB | grab attachment compliance |
| `mat_damping` | `float` | NB | 1/s |
| `mat_friction` | `float` | NB | |
| `mat_restitution` | `float` | NB | |
| `cap_a`,`cap_b` | `vec3` | C | capsule endpoints, world space, at the end of this frame |
| `cap_a_prev`,`cap_b_prev` | `vec3` | C | the same endpoints at the end of the last frame, so a substep can sweep the capsule between them instead of teleporting it |
| `cap_r` | `float` | C | radius; **0 means inactive** |
| `cap_va`,`cap_vb` | `vec3` | C | endpoint velocities, m/s |
| `cap_count` | `int32` | 1 | device-side active count |
| `grab_particle` | `int32` | G | particle index, −1 when unused |
| `grab_local` | `vec3` | G | offset in the pinch frame at grab time |
| `grab_hand` | `int32` | G | which hand slot holds it |
| `grab_lambda` | `float` | G | XPBD multiplier of the grab attachment constraint |
| `grab_count` | `int32` | 1 | device-side active count |
| `hand_pos` | `vec3` | H | pinch point, world space |
| `hand_rot` | `quat` | H | pinch frame orientation |
| `hand_vel` | `vec3` | H | pinch point velocity |
| `hand_active` | `int32` | H | 1 while this slot holds something |
| `contact_count` | `int32` | 1 | diagnostics |
| `body_bad` | `int32` | NB | set by `contain_particles` when a body has gone non-finite or left the world |
| `tet_wsum_rest` | `float` | T | the element's mass term at `F = I`, for `resolvable_softening` |
| `tri_idx` | `vec3i` | — | every body's render triangles, in global indices, for `compute_normals` |
| `params` | `float` | `NUM_PARAMS` | every scalar the kernels read that the user can change: ground, drag, margins, both clamps, the basin, wind turbulence, the simulation clock |
| `vparams` | `vec3` | `NUM_VPARAMS` | gravity and the wind vector |

`params` and `vparams` are why the CUDA graph rule below is keepable: a value
that changes between frames lives in one of them and the captured graph reads
it from the device, where a kernel argument would have been baked in at
capture time and frozen silently.

Colour batches are host-side: `dist_batches: list[tuple[int, int]]` giving
`(offset, count)` per colour, and likewise for bending and tetrahedra.

---

## 6. The maths

### 6.1 XPBD

For a constraint `C(x)` with compliance `α`, at substep `h`:

```
α̃  = α / h²
Δλ = (−C − α̃·λ) / (Σᵢ wᵢ|∇ᵢC|² + α̃)
Δxᵢ = wᵢ · ∇ᵢC · Δλ
λ  += Δλ
```

### 6.2 Distance

`C = |x₀ − x₁| − d`, `∇₀C = n`, `∇₁C = −n` with `n = (x₀−x₁)/|x₀−x₁|`.

### 6.3 Dihedral bending

For an edge `(p₂, p₃)` shared by triangles `(p₂,p₃,p₀)` and `(p₂,p₃,p₁)`, with
`n₁ = (p₂−p₀)×(p₃−p₀)`, `n₂ = (p₃−p₁)×(p₂−p₁)`, `d = n̂₁·n̂₂`, the constraint is
`C = arccos(d) − φ₀`. The gradients are the standard Bridson/Müller ones; the
kernel must guard `|d| → 1` (coplanar triangles give `1/√(1−d²) → ∞`).

### 6.4 Stable Neo-Hookean tetrahedron

Deformation gradient `F = Ds · Dm⁻¹` with `Ds = [x₁−x₀, x₂−x₀, x₃−x₀]`.
Two constraints per tetrahedron (Macklin & Müller 2021):

```
deviatoric:   C_D = √(tr(FᵀF))            α_D = 1/(μ·V)
hydrostatic:  C_H = det(F) − γ,  γ = 1+μ/λ   α_H = 1/(λ·V)
```

with gradients `∂C_D/∂F = F/√(tr(FᵀF))` and `∂C_H/∂F = [f₁×f₂, f₂×f₀, f₀×f₁]`
(columns), both mapped back to particle gradients by `∂F/∂x = Dm⁻ᵀ`:

```
[g₁ g₂ g₃] = (∂C/∂F) · Dm⁻ᵀ ,   g₀ = −(g₁ + g₂ + g₃)
```

The `γ` offset is what makes the rest pose force-free: it cancels the
deviatoric term's non-zero rest force. Dropping it (γ = 1) settles a cube at
0.91–0.96 of its rest volume depending on hardness, measured. With it, the
same cube settled for 3 s holds 0.979 of its rest volume at hardness 0 and
0.999 at hardness 1 — the loss that remains is gravity compacting a soft
body, not the formulation leaking volume.

**An element cannot carry unlimited stiffness.** An XPBD pass lands on the
true compliant force in one iteration only while `α̃` dominates the mass term
`Σ wᵢ|∇ᵢC|²`; below that it degenerates into a hard PBD projection of `C` to
zero. For `C_H` that is harmless — `det F = γ` is a pose an element can reach.
For `C_D` it is ruinous: `√(tr(FᵀF))` is zero only at `F = 0`, so every substep
asks each element to collapse to a point, the edge constraints undo it, and
the pair pumps energy until the body leaves the stage. The crossover is
physics, not tuning: `α̃_D/Σw|∇C|² ≈ 5ρL²/(4μh²)`, i.e. μ ≈ 36 kPa for a 2 cm
element at h = 0.93 ms.

`kernels.resolvable_softening` therefore returns a factor `k ≤ 1` and divides
*both* compliances by it. Scaling μ and λ together is what makes this safe
rather than merely different: it leaves the rest-pose cancellation and γ
exactly intact, so the result is an honest softer material (Young's modulus
scaled by `k`, Poisson ratio untouched) rather than a truncated correction
that would inject energy. Above the ceiling the dial is carried by the tet
edge distance constraints, which project onto a pose that is reachable.
`SolverState.tet_stiffness_ceiling(dt)` reports the limit rather than applying
it in silence — silently softening a material is exactly the kind of thing
that should be reportable. `tests/test_solver.py` reads it; the HUD does not,
so what the dial shows on screen is the *nominal* `E`, not the delivered one.

**On the shipped presets the ceiling binds well inside the dial, and this is
the honest statement of what "hard" then means.** Measured at 90 Hz × 12
substeps (`h` = 0.926 ms), with `k` evaluated per tetrahedron exactly as the
kernel does:

| preset | ceiling (μ) | h = 0.2 | h = 0.4 | h = 0.6 | h = 1.0 |
|---|---|---|---|---|---|
| `soft` (sphere, 1.5 cm cells) | 4.2 kPa | 9% of tets softened | 88% | 100% | 100% |
| `cube` (box, 2.2 cm cells) | 32.4 kPa | 0.1% | 82% | 100% | 100% |

Past the ceiling the *delivered* Neo-Hookean stiffness does not merely stop
rising, it falls: `k` is the smaller of the deviatoric and hydrostatic
headroom and past about hardness 0.4 it is set by λ, which grows faster than
μ as Poisson's ratio climbs, so scaling both by the same `k` — which is what
keeps γ intact — drags μ down with it. On the `soft` preset the median
element delivers an effective `E` of 46 kPa at hardness 0.4 and 10 kPa at
hardness 1.0, against the 12 MPa the HUD names there.

So: μ and λ *are* Young's modulus and Poisson's ratio, and below the ceiling
the feel at a given dial position is `dt`-independent and substep-independent,
which is the claim that matters for the showpiece. What the dial is not,
above about 0.4, is literal: from there on the extra stiffness the user feels
is carried by the tet edge distance constraints, the number on screen is the
material asked for rather than the one the elements resolve, and because the
ceiling goes as `ρL²/h²` that part of the dial does move with the substep
count. The ceiling rises with *coarser* elements and with more substeps — the
2.2 cm cube keeps 99.9% of its elements literal at hardness 0.2 where the
1.5 cm sphere has already softened 9% of its own — so raising
`soft_resolution` moves it the wrong way.

This pair is *why* the hardness dial is a material dial rather than a fudge
factor: `μ` and `λ` come straight from Young's modulus and Poisson's ratio,
and the compliance formulation keeps the feel independent of `dt` and of the
substep count. Read the ceiling above for where the elements stop being able
to deliver what the dial asks for.

### 6.5 Capsule contact

Particle–particle contact is solved **Jacobi**: a particle averages the
corrections its neighbours ask for rather than summing them. Inside a dense
pile a grain has a dozen simultaneous contacts, and adding them up moves it
about twelve times further than any one of them justified — which is why a
settling pile used to fling grains out of the box, and why *more* substeps
made it worse rather than better. A mild over-relaxation buys back some of
what averaging costs, faded in with the contact count so that a particle with
exactly one contact is not over-relaxed at all (there is nothing averaged away
to compensate for, and over-relaxing it separates the pair past the contact,
which `finalize` turns into velocity).

Closest point on segment `ab` to particle `p`:
`t = clamp(dot(p−a, b−a)/dot(b−a, b−a), 0, 1)`, `q = a + t(b−a)`.
Penetration `δ = (r_cap + r_particle) − |p−q|`. If `δ > 0`, push `p` out along
`n̂ = (p−q)/|p−q|` and apply Coulomb friction against the capsule's velocity at
`q` (lerped from `va`, `vb` by `t`), so a hand can drag cloth sideways instead
of only pushing it.

---

## 7. Stability rules

The brief asks for responsiveness and stability before feature count, so these
are requirements, not suggestions:

1. **Substeps over iterations.** 12 substeps × 1 iteration beats 1 × 12.
2. **Compliance, never stiffness multipliers.** Guarantees the feel is
   independent of `dt` and of the substep count.
3. **Positional collision response only.** Contacts move positions and let
   `finalize` derive the velocity. A hand that teleports cannot inject energy.
4. **Correction clamp.** No single constraint may move a particle more than
   `max_correction_ratio × radius` in one substep.
5. **Velocity clamp** at `max_velocity`.
6. **Grab hysteresis** — separate start and release thresholds, plus a hold
   time. Without it the object is dropped and re-grabbed every few frames.
7. **Soft grab attachment.** A grab is a compliant constraint, not a teleport,
   so yanking your hand cannot pull a particle through the floor.
8. **Containment before the hash grid, not after.** `wp.HashGrid.build`
   converts every point to an integer cell index with no range check, so one
   `+inf` — or merely a coordinate somewhere past 1e6 m — is an illegal memory
   access that kills the CUDA context for the whole process, uncatchably.
   `contain_particles` therefore runs *every* step, before the build, snapping
   anything non-finite or outside the world back to rest and flagging its body.
   The `sanity_interval` sweep then consumes that flag and resets the body. A
   guard that only runs every thirtieth step is a third of a second too late.
9. **Colliders fade in.** A hand appears wherever the person's hand is, which
   can be inside the matter.
10. **Held matter does not collide with the hand holding it, and the wrist
    carries the grab.** The grab reaches every particle within its radius of
    the pinch point, and the fingertip capsules sit exactly there, so part of
    what is held starts *inside* a capsule. Letting go sweeps those capsules
    through the held patch, and a particle pushed out of a moving capsule
    leaves at the correction clamp — three radii per substep, 15 m/s — not at
    the finger's speed. Separately, the pinch point is the midpoint of two
    fingertips and moves several centimetres when they open, before the pinch
    value has dropped at all. Measured together: a soft ball let go by a
    perfectly still wrist left at 0.8–0.9 m/s sideways. So `FLAG_GRABBED`
    particles skip capsule contact, and while a slot holds something its
    anchor moves with the wrist rather than the fingertips, and its twist
    with the *palm* frame (wrist, index MCP, pinky MCP) rather than the pinch
    frame — the pinch frame turns whenever the fingers move, and with a few
    hundred particles held deep in a ball that turn was a whip, 5 m/s from
    fingers merely opening. Twisting a held body is done with the wrist,
    which is how a person does it. Now the same release drags the held
    matter at 0.000 m/s and throws it 1.2 mm.
    `tests/test_solver.py` holds it there.
11. **A capsule may not push faster than it moves.** The matter *next to* a
    held patch was still being flung: fingers opening at 0.3 m/s ejected free
    particles at 2.9 m/s, because a particle that begins a substep well
    inside a capsule was pushed to its surface in one go — up to three radii,
    15 m/s — and `finalize` read that as velocity. The capsule correction is
    now bounded per substep by twice the capsule's own speed at the contact
    point, with a small settling rate (a tenth of a radius per substep) so a
    still hand with matter resting inside it still clears itself within a
    frame or two. This is the same shape of fix as the basin wall: a speed
    limit, because a distance clamp is still a speed.

---

## 7a. Determinism

`AppConfig.lockstep` advances exactly one physics step per frame off a virtual
clock instead of following the wall clock. `--headless` and `--benchmark` turn
it on from the command line (`config_from_args`); the dataclass field itself
defaults to `False`, so a caller building an `AppConfig` in code has to ask
for it — which is why every smoke test passes it explicitly. A headless run
finishes as fast as the GPU allows, so on the wall clock
600 frames is two seconds of simulation on one machine and six on another;
in lockstep it is always `600 / rate_hz` seconds. Two identical headless runs
are bit-identical, which is what makes a captured frame comparable and the
smoke test meaningful.

The solver is deterministic by construction: graph colouring means no
constraint kernel ever races, and `tests/test_solver.py` checks that the CUDA
graph path and the plain path agree bitwise.

---

## 8. Controls

| input | effect |
|---|---|
| mouse wheel / `[` `]` (also `-` `=`, `down` `up`) | hardness down / up; held, it ramps |
| drag with right mouse | orbit the camera |
| ctrl + mouse wheel | zoom |
| `1`–`5` | switch preset (cloth, banner, soft, cube, grain) |
| `R` | reset the scene |
| `H` | toggle the HUD |
| `W` | toggle the webcam inset |
| `K` | toggle the 3D hand skeleton |
| `G` | toggle wireframe |
| `P` | pause / resume physics |
| `.` | single-step while paused |
| `F` | toggle wind |
| `D` | run the choreographed demonstration (`fctx.demo`): grab, lift, sweep the dial while holding, let go; loops until pressed again. On a camera source it drives only the dial |
| `A` | sweep the hardness dial automatically, for demos and recordings |
| `0` | jump the dial to its soft end |
| `F9` | start / stop recording the tracked hands to a `.fhr` file |
| `F11` | fullscreen |
| `F12` | screenshot |
| `Esc` | quit |
| mouse, `space` / click | (synthetic hand) steer and pinch |
| `Q` / `E`, `C` | (synthetic hand) depth, curl |

`render/hud.py`'s `CONTROL_HINT` is the on-screen copy of this table, and it
has to stay a copy: for anyone watching a recording it is the only place
these keys exist. `tests/test_render.py` checks that every key above appears
in it.

---

## 9. Testing

`tests/` runs without a camera and, where marked, without a GPU.

* `test_types.py` — `BodyData.validate` catches malformed bodies.
* `test_material.py` — the dial is monotonic, compliance is positive and
  finite across the whole range, and the endpoints match the declared
  constants.
* `test_coloring.py` — no two constraints of the same colour share a particle,
  for every body the builders can produce.
* `test_bodies.py` — tetrahedra have positive volume, surfaces are closed and
  outward-wound, cloth rest lengths match the grid spacing.
* `test_filters.py` — the One-Euro filter is stable, converges, and passes a
  step input through faster than it passes noise.
* `test_projection.py` — image→world is monotonic and clamps as documented.
* `test_hands.py` — track ids stay with their hand when the list reorders or
  a hand drops out and returns, a lost hand coasts and then disappears, the
  pinch hysteresis does not chatter, and a record/replay round trip
  reproduces the poses.
* `test_app.py` — the wiring no single subsystem can see on its own: which
  hand is in which solver slot, the fixed-timestep clock, and which half of a
  preset survives a runtime switch.
* `test_render.py` *(GPU)* — every shader compiles; a headless frame is
  neither black nor white; the pipeline survives absurd window sizes, two
  contexts, rebinding and release; the HUD fits every window shape; grains
  keep the shadows they cast on each other.
* `test_solver.py` *(GPU)* — a hanging cloth reaches equilibrium without NaN; a
  soft body dropped on the floor conserves volume within tolerance; hardness
  extremes are both stable; results are identical with and without the CUDA
  graph.
* `test_demo.py` — the choreographed demonstration is a well-formed timeline:
  no field jumps between physics steps, the preset changes exactly on its
  keyframe, captions carry forward, bad timelines are refused.
* `test_settings.py` — the configuration file round-trips exactly for every
  preset, sections override field by field, typos fail loudly and name the
  key, and the CLI layers file then flags with `--preset` winning.
* `test_resilience.py` *(GPU)* — a camera that dies mid-session is replaced
  by the synthetic hand and taken back when it returns; attract mode starts
  with nobody there and stops within `ATTRACT_WAKE_FRAMES` of a hand
  appearing, survives the camera going and coming back, and never takes a
  synthetic-only run; a resilient frame loop recovers up to
  `MAX_CONSECUTIVE_FAILURES` times and then stops.
* `test_smoke.py` *(GPU)* — the whole application, headless, on the synthetic
  hand: it starts and shuts down cleanly, two identical runs are bit-identical,
  every preset runs without throwing matter off the stage, the dial reaches the
  solver, a screenshot comes out with real contrast and colour, a runtime
  preset switch rebuilds everything, the camera keeps both the matter and the
  hand's reach in frame, a missing camera falls back instead of failing, and a
  record/replay round trip reproduces the hands.

---

## 9a. The demonstration

`fctx.demo.Choreography` is a keyframed timeline in the synthetic hand's own
control space — pointer, depth, pinch, curl — plus the dial and the preset,
interpolated with a smoothstep so nothing jumps. The application applies it by
overwriting the same `ControlState` fields the mouse and keyboard write, so
everything downstream (the synthetic source, the dial smoothing, the grip
hysteresis, the HUD) behaves exactly as it does for a person. On a camera
source the hand is the person's and only the dial follows the script.

The shipped choreography is two acts: a sheet of cotton is pinched, lifted,
walked from hardness 2% to 97% and back while held (measured: 82 particles
held throughout; contacts with the hand fall from ~250 to ~55 as it stiffens,
because a plate touches a palm less than a drape does), swung, and released;
then a soft body is grabbed by its cap, lifted, walked from jelly to rigid and
back while hanging, dropped rigid so it bounces, and melted where it lies.

Keyframes are in seconds of simulation time, so under `lockstep` a rendered
video (`tools/render_showcase.py`, H.264 through OpenCV, 60 fps from the 90 Hz
physics by accumulated time) and a live run agree to the frame. The
synthetic source's autopilot is disabled for the duration: holding still is
the whole point of the showpiece, and the autopilot reads a cue that does not
change as idleness.

---

## 9b. Running unattended

An installation is left alone for a day at a time, so `MatterStudio` has
three behaviours that only matter when nobody is watching. All three are
exercised by `tests/test_resilience.py` with a scripted camera standing in for
the real one, because the failures they guard against live in the wiring
between source, tracker, demo and frame loop.

**Camera hot-plug.** The live camera is `app.live_source`; whatever currently
drives the matter is `app.source`, and they are not always the same object.
`CameraUnavailable` from a poll hands the stage to the synthetic source at
once (`_camera_lost`), and a daemon thread retries `create_source` every
`CAMERA_RETRY = 3.0` s (`_poll_camera_retry`) — off the frame loop, because
probing an absent device takes a third of a second and would be a visible
hitch every retry. When it succeeds the tracker is rebuilt, so no stale track
ids or filter state cross the gap. The same path covers a camera that was
missing at start-up.

**Attract mode.** With `idle_demo > 0` and a camera wanted, `idle_demo`
seconds without a hand in any live frame starts the choreographed
demonstration (§9a) on the synthetic hand (`_poll_attract`). The camera keeps
being polled — that is what `_update_tracking` does with `live_source` when
it is not `source` — and `ATTRACT_WAKE_FRAMES = 3` consecutive frames with a
hand in them give the stage back, stop the demonstration, and rebuild the
tracker. A camera lost in an empty room does not stop the demonstration and
a camera returning to one does not start real tracking; only a person does.
A run configured for the synthetic source ignores `idle_demo`, since nothing
could ever wake it and the mouse-driven hand would be lost to the demo.

**A frame that raises.** With `resilient` set, an exception in the frame loop
is logged with its traceback (`log`, to `log_file` when there is one), the
solver and grips are reset, and the loop continues. `MAX_CONSECUTIVE_FAILURES
= 30` failures in a row are not a bad frame but a broken installation, and
the process exits so that a supervisor (`run_kiosk.bat`) can restart it. A
good frame clears the count. Without `resilient` the first exception
propagates, which is what a developer wants.

**A scheduled restart.** `max_uptime` hours after start, `_restart_due`
ends the run cleanly -- but only once no hand has been seen for `idle_demo`
seconds (`RESTART_IDLE` when there is no attract mode), or at once when
there is no camera to see one. The supervisor loop starts a fresh process.
This is hygiene, not a fix for something known to be broken: a 45-minute
headless soak (`tools/soak.py`) showed dedicated GPU memory flat and the
process working set creeping by about 57 bytes per drawn frame -- per frame,
not per second (slowing the loop down changes nothing), outside the Python
heap (`tracemalloc` is flat), absent when drawing is skipped, a third of it
the GPU timer query and none of it physics. At a 60 Hz display that is
12 MiB/h; the unthrottled soak at 230 fps read 52 MiB/h. Twelve hours of
either is a few hundred megabytes, which a restart in an empty room makes
irrelevant.

`--kiosk` is the bundle: fullscreen, `resilient`, `idle_demo = 20`,
`max_uptime = 12`.

---

## 10. Known limits

These are measured, not suspected.

* **Granular friction is weak, and the dial barely moves it.** Position-based
  friction is proportional to penetration depth, which goes to zero at rest,
  so a pile never actually reaches an angle of repose: unconfined, the shipped
  24 000 grains read 8.2° after 3 s, 6.4° after 6 s and 4.8° after 12 s, still
  spreading, against a real material's 30-odd. Friction changes this by less
  than half a degree over the whole dial (μ = 0.95 and μ = 0.35 agree to
  within 0.5° at every sample, and the 95th-percentile radius to within
  0.02 m) — it only shows in the few outermost grains, whose reach goes from
  0.48 m at μ = 0.95 to 0.61 m at μ = 0.35. That is the mechanism being
  honest about itself: a friction that vanishes at rest cannot hold a heap up.
  It is why the granular scene ships with a basin. Fixing it properly means
  resolving friction in the velocity pass against an accumulated contact
  frame.
* **The hash grid is rebuilt once per step, not per substep.** `HashGrid.build`
  cannot be called from inside a CUDA graph capture. Contacts are therefore
  found against start-of-step positions and a fast pile finds some of them a
  step late and already deep. The contact correction is bounded by the contact
  sphere so a late contact cannot pay off its whole debt in one substep.
* **The skinned surface is a blend of affine maps, not a true subdivision
  surface.** At rest it is exact (0.12 mm of radial spread on a 130 mm
  sphere, against 6.0 mm for the lattice underneath) and blending over eight
  elements keeps it smooth under deformation, but a very sharp local dent —
  one fingertip into jelly — is still resolved at the lattice's resolution
  underneath, not the skin's.
* **A handful of tetrahedra invert transiently under contact.** Up to 8 of
  24725 on the shipped sphere after twelve seconds at hardness 1.0, with the
  body at rest, 0.03 m/s and 99.6% of its volume. They do not feed themselves,
  and the hydrostatic constraint recovers most of them — at hardness 0 the
  count peaks at 7 and is back to 0 by three seconds.
  `tests/test_solver.py` therefore bounds the fraction and its growth rather
  than asserting zero: zero held at exactly one pair of tuning constants and
  was a coincidence, not a property.
* **The physics surface sits about 3.6 mm inside the drawn one** on the
  shipped sphere, because the lattice only takes a gentle fit. A hand makes
  contact a few millimetres before the two surfaces visibly touch — well
  inside the 9–19 mm radius of the capsule doing the touching, so it does not
  read as a gap.
* **A variable `dt` recaptures the CUDA graph every frame**, costing 6–20 ms
  per step. The fixed-timestep clock never does this; a caller that drives
  `step()` from a wall clock would.
* **The hard end of the dial is not a literal material** on either shipped
  soft preset: see §6.4. The visible consequence is at the very top. Settled
  on the floor for 6 s, the `cube` preset is 1.5% wider than its rest shape at
  hardness 0.55 and 22.7% wider at hardness 1.0, where the HUD reads
  `NEAR RIGID  E = 12.0MPa`. It is a settled state, not a transient: the width
  is 1.2272 of rest at every second from t = 2 s to t = 8 s and Σ|v|²/2 has
  decayed to 4.7e-4 — but a near-rigid body should not settle
  permanently fatter than it started, and "the same shape stops behaving like
  jelly and starts behaving like rubber" is not true of the last fifth of the
  dial's travel.
