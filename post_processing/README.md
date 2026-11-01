# Post-processing: dialogue figures

Scripts that turn one recorded dialogue session into figures and plotter drawings.
The examples use Snapper's SIGGRAPH session `session-20260722-114251`.

Run everything from the repo root. Figures and caches go in `outputs/`, which is git-ignored.

```bash
SESSION="/Volumes/Cohab2024/BFF/by-phase/4-SIGGRAPH-2026-07/KEY__session-20260722-114251__snapper"
MODEL=/Volumes/Work/Projects/bff/models/gemma-4-E2B-it
HM=/opt/homebrew/anaconda3/envs/housemachine/bin/python3
```

## Requirements

| Needed for | Requirement |
|---|---|
| Most scripts | `python3` from the anaconda base env (`numpy`, `scipy`, `scikit-learn`, `torch`, `transformers`) |
| `plotter_drawing.py` | `$HM`, the `housemachine` conda env: it has `HersheyFonts`, and the script needs only that and `numpy` |
| Replaying sessions | `google/gemma-4-E2B-it` (bf16) at `$MODEL`. About 14.5 GB peak on MPS |
| `--png` | `inkscape` (`rsvg-convert` also works, except for `plotter_drawing.py`) |

A `SESSION` folder needs `transcript/session-complete.jsonl`, or `session.jsonl` as a fallback.

## The scripts

| Script | Makes |
|---|---|
| `conversation_graph.py` | 2D dialogue graph with the prompt payload per turn |
| `graph3d.py` | 3D dialogue graph: projected render, interactive viewer, OBJ, JSON |
| `plotter_drawing.py` | Black-ink pen-plotter drawing, from `graph3d.py`'s or `multi_session.py`'s JSON |
| `multi_session.py` | Several sessions in one shared space, for `plotter_drawing.py` |
| `conversation_map.py` | A fixed map of all Snapper exchanges, and a session's turns placed on it |
| `map_compare.py`, `map_report.py` | Compare feature vectors for that map |
| `activation_graph.py` | 2D graph with six other depths and a depth profile; writes the activation cache |
| `similarity_grid.py` | Similarity matrix at each of the 36 depths |
| `attention_flow.py` | Where each reply's attention went; writes the attention cache |
| `surprise_ribbons.py` | The conversation as text, shaded by token surprise |
| `dialogue_poster.py` | 22x30 in poster combining text, attention and surprise |

## 1. Build the caches

The figures read two replays of the session through Gemma 4. Each takes about a minute, and
needs doing once per session and system prompt.

```bash
python3 post_processing/activation_graph.py "$SESSION" --model $MODEL --system-prompt performance-script.json-02 -o outputs/snapper-activation-graph.svg --png
python3 post_processing/attention_flow.py "$SESSION" --attn outputs/snapper-attention.json --model $MODEL --system-prompt performance-script.json-02 -o outputs/snapper-attention.svg --png
```

- `--system-prompt` matters. The logs record only the config prompt, but `chat-manager.py` replaces
  it with the performance script's Default scene. `performance-script.json-02` is inferred, as the
  best fit to the logged replies, not confirmed. Performance scripts are git-ignored and per-machine.
- Rebuilt values differ slightly from the caches in `outputs/`: token surprise by about 0.1 bits on
  average (1.2 at most) and attention shares by at most 0.4 percentage points. Two rebuilds in a row
  matched each other exactly, so this is a shift since the caches were made, not noise between runs.
  The cause is not established.

## 2. The current figures

### Plotter drawing

The drawing has two steps: `graph3d.py` makes the layout, and `plotter_drawing.py` draws it.

```bash
python3 post_processing/graph3d.py "$SESSION" --acts outputs/snapper-activation-graph.acts.npz -o outputs/snapper-graph3d.svg --viewer --obj --png
$HM post_processing/plotter_drawing.py --graph outputs/snapper-graph3d.json --attn outputs/snapper-attention.json -o outputs/snapper-plotter-default.svg --png
```

This places nodes by Gemma's hidden state at layer 18, in context, reduced to 3D by MDS and seen at
azimuth 30, elevation 20, camera 4. The drawing shows:

- circles sized by spoken duration: single rings for Robert, double rings for Snapper
- the full text of each utterance
- turn numbers and arrowheads along the trail
- dashed arcs to and from a stack of the utterances that never entered the model
- a title in the lower right, with the conversation's date and time beneath it

#### Changing the camera

The 3D layout stays the same; only the view changes, so there is no need to rerun `graph3d.py` or the
model. Find a view in `outputs/snapper-graph3d.html` (drag to rotate, scroll for perspective), copy
the flags at the top of the page, and add them to the drawing command. Give each view its own `-o`,
because that name also sets the PNG and sidecar names.

```bash
$HM post_processing/plotter_drawing.py --graph outputs/snapper-graph3d.json --attn outputs/snapper-attention.json --azimuth 147 --elevation 17 --camera 6.5 -o outputs/snapper-plotter-az147-el17-cam6.5.svg --png
```

The same flags work on any layout. For the TF-IDF drawing, use its JSON and viewer
(`outputs/snapper-graph3d-tfidf.html`):

```bash
$HM post_processing/plotter_drawing.py --graph outputs/snapper-graph3d-tfidf.json --attn outputs/snapper-attention.json --azimuth 147 --elevation 17 --camera 6.5 -o outputs/snapper-plotter-tfidf-az147-el17-cam6.5.svg --png
```

For the graph itself rather than the pen drawing, pass them to `graph3d.py`. This re-runs the 3D
layout, which gives the same result each time:

```bash
python3 post_processing/graph3d.py "$SESSION" --acts outputs/snapper-activation-graph.acts.npz --azimuth 147 --elevation 17 --camera 6.5 -o outputs/snapper-graph3d-az147-el17-cam6.5.svg --png
```

- `--azimuth` turns the graph about its vertical axis, in degrees.
- `--elevation` raises the viewpoint, in degrees.
- `--camera` is the distance in graph radii. Smaller is stronger perspective, `0` is orthographic, and
  anything else must be above 1.5.
- Any flag left out comes from the JSON, which has the view `graph3d.py` was run with (30, 20, 4).

The drawing's options:

| Option | Effect |
|---|---|
| `--surprise` | The trail wanders with token surprise |
| `--attention` | Adds attention strands (`--strand-pct`, default 0.5) |
| `--knots` | Scribbled knots replace the circles |
| `--excerpt [N]`, `--speaker` | Opening N words, or the speaker's name, instead of full text |
| `--no-sequence`, `--no-date` | Remove the turn numbers and arrows, or the date line |
| `--leader solid`, `--leader-gap` | Text leaders are dotted by default, with dots 2.5 mm apart. Each dot is a pen drop |
| `--paper WxH` | Paper size in inches, default `30x22` |
| `--title`, `--title-height`, `--seed`, `--stroke-mm` | Title text and size, random seed, pen weight |

The dense drawing: `--surprise --attention --knots --speaker -o outputs/snapper-plotter.svg`.

30x22 in needs the A0 machine. `--paper 14x11 --strand-pct 1` fits the AxiDraw A3. Check a drawing
before plotting; this reads the file and does not touch the plotter:

```bash
$HM /Volumes/Work/Projects/housemachine-current/code/machine_plotting/send_svg_axidraw.py outputs/snapper-plotter-default.svg --inspect --overdraw
```

### Several conversations in one space

`multi_session.py` replays each session, then reduces all their utterances together, so positions can
be compared across recordings. It writes the same files as `graph3d.py` and the attention cache, so
`plotter_drawing.py` needs no extra option.

```bash
B=/Volumes/Cohab2024/BFF/by-phase/4-SIGGRAPH-2026-07
python3 post_processing/multi_session.py $B/KEY__session-20260720-195543__snapper $B/KEY__session-20260720-204648__snapper $B/KEY__session-20260722-114251__snapper --model $MODEL --system-prompt performance-script.json-02 --names "i feel dead inside" "apple as mirror" "mccarthy and lovelace" -o outputs/snapper-multi.json
$HM post_processing/plotter_drawing.py --graph outputs/snapper-multi.json --attn outputs/snapper-multi-attention.json -o outputs/snapper-multi-plotter.svg --png
```

Other sheet sizes, with six-word excerpts. `--paper` is width by height, so these are landscape, and
each needs its own `-o`. Check the result: on a smaller sheet, labels can still sit tight or cross lines.

```bash
$HM post_processing/plotter_drawing.py --graph outputs/snapper-multi.json --attn outputs/snapper-multi-attention.json --paper 17x14 --excerpt 6 -o outputs/snapper-multi-plotter-14x17-excerpt.svg --png
$HM post_processing/plotter_drawing.py --graph outputs/snapper-multi.json --attn outputs/snapper-multi-attention.json --paper 17x11 --excerpt 6 -o outputs/snapper-multi-plotter-11x17-excerpt.svg --png
```

- Sessions are ordered as given. Turns are lettered by session (A12, B3).
- Each session has its own stack at the left. Sessions are not connected to each other.
- `--session-ticks` marks each session's trail with 0, 1 or 2 cross-strokes, shown in the key. Off by default.
- Labels are full text by default, which gets crowded with this many utterances. `--excerpt N` gives
  the opening N words instead, and a larger `--paper` gives the text more room.
- All sessions are replayed with one system prompt, and scene switches inside a session are not
  reproduced, so choose sessions without them. The prompt is an assumption for each session.
- `outputs/snapper-multi.html` is the viewer for finding a camera. Rerunning with more sessions reuses
  the replays cached in `outputs/multi/`.
- The last run took about 15 minutes per session, far longer than the minute a replay took in
  September. The cause has not been found.

### Flat layouts

`--flat` on `graph3d.py` or `multi_session.py` runs MDS to two dimensions instead of three, and the
drawing then reads it straight on. It is different from pointing the camera at the 3D form with
`--azimuth 0 --elevation 0 --camera 0`, which only drops the third axis. No model replay is needed
once the caches exist. Pass `--azimuth` and the like and the plane simply tilts, so leave them out.

```bash
python3 post_processing/graph3d.py "$SESSION" --acts outputs/snapper-activation-graph.acts.npz --flat -o outputs/snapper-flat.svg --png
$HM post_processing/plotter_drawing.py --graph outputs/snapper-flat.json --attn outputs/snapper-attention.json -o outputs/snapper-plotter-flat.svg --png

B=/Volumes/Cohab2024/BFF/by-phase/4-SIGGRAPH-2026-07
python3 post_processing/multi_session.py $B/KEY__session-20260720-195543__snapper $B/KEY__session-20260720-204648__snapper $B/KEY__session-20260722-114251__snapper --model $MODEL --system-prompt performance-script.json-02 --names "i feel dead inside" "apple as mirror" "mccarthy and lovelace" --flat -o outputs/snapper-multi-flat.json
$HM post_processing/plotter_drawing.py --graph outputs/snapper-multi-flat.json --attn outputs/snapper-multi-flat-attention.json --paper 17x11 --excerpt 6 -o outputs/snapper-multi-flat-plotter-11x17-excerpt.svg --png
```

The three conversations on the full 30x22 sheet, with full text. Add `--excerpt 6` for shorter labels:

```bash
$HM post_processing/plotter_drawing.py --graph outputs/snapper-multi-flat.json --attn outputs/snapper-multi-flat-attention.json -o outputs/snapper-multi-flat-plotter.svg --png
```

`multi_session.py --space` takes the same choices as the other scripts. `activation` is the default.
`tfidf` places all the sessions by word overlap, and `standalone` runs each utterance alone through the
model (cached in `outputs/multi/`). Both give every utterance a position, so there are no stacks.
`--flat` works with each. `--model` is still required, but tfidf uses it only for missing session caches.

```bash
python3 post_processing/multi_session.py $B/KEY__session-20260720-195543__snapper $B/KEY__session-20260720-204648__snapper $B/KEY__session-20260722-114251__snapper --model $MODEL --system-prompt performance-script.json-02 --names "i feel dead inside" "apple as mirror" "mccarthy and lovelace" --space tfidf --flat -o outputs/snapper-multi-tfidf-flat.json
$HM post_processing/plotter_drawing.py --graph outputs/snapper-multi-tfidf-flat.json --attn outputs/snapper-multi-tfidf-flat-attention.json -o outputs/snapper-multi-tfidf-flat-plotter.svg --png
```

### Fixed conversation map

One map of all Snapper's conversations, so every plot shares its coordinates. Needs `map_compare.py
corpus minilm summary` to have run, and uses the `bff` conda env, which has `umap-learn`.

```bash
BFF=/opt/homebrew/anaconda3/envs/bff/bin/python
$BFF post_processing/conversation_map.py fit
$BFF post_processing/conversation_map.py place "$SESSION" -o outputs/snapper-exchange-map.json
$HM post_processing/plotter_drawing.py --graph outputs/snapper-exchange-map.json -o outputs/snapper-exchange-map-plotter-full.svg --png
```

Several sessions go on the same map. Turns are lettered by session, and each has its own stack:

```bash
B=/Volumes/Cohab2024/BFF/by-phase/4-SIGGRAPH-2026-07
$BFF post_processing/conversation_map.py place $B/KEY__session-20260720-195543__snapper $B/KEY__session-20260720-204648__snapper $B/KEY__session-20260722-114251__snapper --names "i feel dead inside" "apple as mirror" "mccarthy and lovelace" -o outputs/snapper-multi-exchange-map.json
$HM post_processing/plotter_drawing.py --graph outputs/snapper-multi-exchange-map.json --excerpt 6 -o outputs/snapper-multi-exchange-map-plotter-excerpt.svg --png
```

`fit` is done once and saved in `outputs/map/`; refitting changes the map. `place` puts each turn on
it separately, so the path steps between question and reply, and resets stay in the stack. The drawing
scales to the whole map and shows every exchange as a faint mark with region names (`--no-backdrop`
removes them, and `--mark slash` or `dash` changes the symbol; the default `plus` costs two pen drops per
exchange). `--no-marks` leaves out the marks but keeps the region names. `--attn` is only needed for `--surprise` and `--attention`.

### TF-IDF drawing

The same drawing, placed by word overlap instead of by the model. No model replay is needed for the
layout, but it still reads the attention cache for the drawing.

```bash
python3 post_processing/graph3d.py "$SESSION" --space tfidf -o outputs/snapper-graph3d-tfidf.svg --viewer --obj --png
$HM post_processing/plotter_drawing.py --graph outputs/snapper-graph3d-tfidf.json --attn outputs/snapper-attention.json -o outputs/snapper-plotter-tfidf.svg --png
```

Every utterance has a position in this space, so there is no stack: the reset request and cue sit in
the form, resets are drawn as dashed arcs between them, and the text is the re-transcription rather
than what the model received. The layout is lexical, so utterances sit close because they share
vocabulary. It keeps less of its distance structure than the activation layout: rho 0.71 in 3D,
against 0.84.

### 2D dialogue graphs

```bash
python3 post_processing/conversation_graph.py "$SESSION" --acts outputs/snapper-activation-graph.acts.npz -o outputs/snapper-dialogue-graph-activation.svg --png
python3 post_processing/conversation_graph.py "$SESSION" --space standalone --system-prompt performance-script.json-02 --model $MODEL -o outputs/snapper-dialogue-graph-standalone.svg --png
python3 post_processing/conversation_graph.py "$SESSION" --space tfidf -o outputs/snapper-dialogue-graph-tfidf.svg --png
```

`--space` picks what places the nodes, and `graph3d.py` takes the same choices:

- `activation` is the default: the hidden state in context, at `--layer` (default 18).
- `standalone` runs each utterance alone, with no prompt or history.
- `tfidf` uses word overlap.

### Other figures

```bash
python3 post_processing/similarity_grid.py "$SESSION" --acts outputs/snapper-activation-graph.acts.npz --model $MODEL -o outputs/snapper-similarity-grid.svg --png
python3 post_processing/surprise_ribbons.py "$SESSION" --attn outputs/snapper-attention.json --forensics outputs/snapper-prompt-forensics.json -o outputs/snapper-surprise.svg --png
python3 post_processing/dialogue_poster.py "$SESSION" --acts outputs/snapper-activation-graph.acts.npz --attn outputs/snapper-attention.json -o outputs/snapper-poster.svg --png
```

`--forensics` is optional. It adds the panel comparing candidate system prompts, and no script's
command line writes that file: it came from a call to `activation_graph.score_prompts()`. Without
the flag, `surprise_ribbons.py` leaves the panel out.

## Using another session

Some text is specific to this session:

- the "SNAPPER × ROBERT" heading in the 2D and 3D figures
- the `chat-manager.py` version in the activation figure's footer
- "temperature 0.7" in the surprise figure's footer
- `graph3d.py` always labels the dog "SNAPPER", so a Helper session needs that changing
- `plotter_drawing.py`'s default title (use `--title`)

Check the system prompt too: pass the performance script that session ran.
