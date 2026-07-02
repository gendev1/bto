# Soccer Plays & Patterns — Full Reference

A catalogue of on-pitch plays, structures, and movements, grouped by phase of play. The "Detectable" column notes how hard each is to catch from single-camera broadcast coordinates (your overlay project): **Easy** = geometry on positions, **Medium** = needs ball + timing, **Hard** = needs off-ball intent, 3D, or limb-level data.

---

## 1. In Possession — Build-up (own third)

| Play                                  | What it is                                                 | Detectable |
| ------------------------------------- | ---------------------------------------------------------- | ---------- |
| Playing out from the back             | GK + defenders pass short to advance rather than long-kick | Medium     |
| Goalkeeper as sweeper-keeper          | GK positions high to act as an extra outfield passer       | Easy       |
| Splitting center-backs                | CBs spread wide to stretch the first press line            | Easy       |
| Dropping pivot / La Salida Lavolpiana | Defensive mid drops between CBs to make a back three       | Medium     |
| Third-man run                         | A → B, B lays off to a third player arriving into space    | Hard       |
| Switch of play                        | Long diagonal to change the point of attack                | Medium     |
| Bounce pass (wall pass setup)         | Quick one-touch to reset and change angle                  | Medium     |

## 2. In Possession — Progression (middle third)

| Play                              | What it is                                                | Detectable |
| --------------------------------- | --------------------------------------------------------- | ---------- |
| Triangle passing                  | Three players forming passing triangles for short options | Easy       |
| Give-and-go / one-two / wall pass | Pass and immediately receive the return past a defender   | Medium     |
| Overlap                           | Fullback runs outside the winger to stretch wide          | Easy       |
| Underlap                          | Fullback/mid runs inside the winger into the half-space   | Easy       |
| Third-man combination             | Same as third-man run, in midfield                        | Hard       |
| Line-breaking pass                | Vertical pass that eliminates a defensive line            | Medium     |
| Half-space occupation             | Player positions in the channel between wing and center   | Easy       |
| Rotations (positional play)       | Players swap zones to disorganize markers                 | Medium     |
| Progressive carry / dribble       | Ball carried forward past opponents                       | Medium     |
| Diamond / box midfield shape      | Structural passing shape in central areas                 | Easy       |

## 3. In Possession — Final Third / Chance Creation

| Play                                   | What it is                                        | Detectable |
| -------------------------------------- | ------------------------------------------------- | ---------- |
| Cutback                                | Ball pulled back from byline to arriving attacker | Medium     |
| Cross (in/out swinger, low, deep)      | Delivery into the box                             | Medium     |
| Through ball                           | Pass split behind the defensive line              | Medium     |
| Overload one side, switch to weak side | Draw defenders, then swing to the free flank      | Medium     |
| Combination play / 1-2s in the box     | Quick short exchanges to unlock a packed defense  | Hard       |
| Isolation (1v1 iso)                    | Deliberately leaving a winger 1v1 on a fullback   | Easy       |
| Runs in behind (blindside / near-far)  | Timed runs behind or across defenders             | Hard       |
| Decoy run                              | Run to drag a defender and open space for another | Hard       |
| Cut inside & shoot                     | Wide player cuts onto stronger foot to shoot      | Medium     |
| Overlap-to-cross                       | Overlap creating the crossing lane                | Medium     |

## 4. Out of Possession — Defensive Structure

| Play                             | What it is                                       | Detectable |
| -------------------------------- | ------------------------------------------------ | ---------- |
| Compact block (low / mid / high) | Whole team's vertical + horizontal compactness   | Easy       |
| Back four / back five shape      | Number and spacing of the last line              | Easy       |
| Defensive line height            | How high the last line holds                     | Easy       |
| Man-marking                      | Each defender tracks a specific opponent         | Medium     |
| Zonal marking                    | Defenders cover zones, not men                   | Medium     |
| Hybrid marking                   | Mix of the two, often on set pieces              | Hard       |
| Offside trap                     | Line steps up together to catch a runner offside | Medium     |
| Covering / balance               | Weak-side defenders tuck in to cover             | Easy       |
| Screening passing lanes          | Body-positioning to block a lane (cover shadow)  | Hard       |
| 1v1 defending / jockeying        | Defender delays attacker without diving in       | Easy       |

## 5. Out of Possession — Pressing

| Play                             | What it is                                                 | Detectable |
| -------------------------------- | ---------------------------------------------------------- | ---------- |
| High press                       | Pressuring in the opponent's build-up third                | Easy       |
| Gegenpress / counter-press       | Immediate press on losing the ball                         | Medium     |
| Pressing trap                    | Inviting a pass, then swarming the receiver                | Hard       |
| Pressing triggers                | Coordinated press on a specific cue (back pass, bad touch) | Hard       |
| N-v-N local overloads (2v2, 3v3) | Matched or numerical pressure in a zone                    | Easy       |
| Cover shadow press               | Pressing while blocking the pass behind you                | Hard       |
| Forcing wide / forcing inside    | Angling the press to steer the ball a direction            | Medium     |
| Pressing the back pass           | Collective jump when the ball goes backward                | Medium     |

## 6. Transitions

| Play                               | What it is                                       | Detectable |
| ---------------------------------- | ------------------------------------------------ | ---------- |
| Counter-attack                     | Fast attack immediately after winning the ball   | Medium     |
| Counter-press (rest defense)       | Structure kept during attack to defend turnovers | Easy       |
| Fast break / direct transition     | Vertical ball forward before defense sets        | Medium     |
| Recovery runs                      | Sprinting back to reform the defensive block     | Easy       |
| Delaying / fouling to stop a break | Tactical foul to reset                           | Hard       |

## 7. Set Pieces (dead-ball)

| Play                              | What it is                                  | Detectable |
| --------------------------------- | ------------------------------------------- | ---------- |
| Corner — near/far post routine    | Designed delivery + runs                    | Medium     |
| Corner — short corner             | Two-player combo to change the angle        | Medium     |
| Corner — blocking/screen (pick)   | Legal-ish screens to free a header          | Hard       |
| Free kick — direct shot           | Shot straight at goal                       | Easy       |
| Free kick — wall + dummy runners  | Runners over the ball to disguise the taker | Hard       |
| Free kick — worked routine        | Short pass to open a new angle              | Medium     |
| Throw-in — long throw             | Delivered into the box like a set piece     | Medium     |
| Throw-in — short combination      | Quick 1-2 to retain and advance             | Medium     |
| Penalty                           | Spot kick                                   | Easy       |
| Kickoff routine                   | Designed first sequence from center         | Easy       |
| Zonal vs man defending on corners | How the defense organizes the box           | Medium     |

## 8. Individual Actions / Skills (context for overlays)

Nutmeg, step-over, Cruyff turn, drag-back, feint, take-on/beat-your-man, shielding the ball, first-touch out of pressure, dummy/let-it-run, backheel, chip, volley, header (attacking/defensive), sliding tackle, block/interception, clearance, last-ditch block.
_(Mostly Hard to auto-detect — these are micro-actions, not spatial patterns.)_

---

## Notes for your overlay build

- The **Easy** rows are your v1 targets — they're pure geometry on `(player, team, x, y, t)` and don't need the ball: formation shape, line height, compactness, isolation, overlap/underlap, NvN overloads, triangles, offside line (approx).
- **Medium** rows unlock once ball tracking (C9) is working: passes, through balls, cutbacks, counters, back-pass pressing.
- **Hard** rows need off-ball intent, cover-shadow reasoning, or 3D — realistically v2+ or out of scope. Don't promise these in a demo.
- The three you already spotted map here as: "1-1 defense" = **1v1 defending / man-marking**; "3-3 triangle passing" = **triangle passing** + **NvN overload**; "pass back" = **back pass** (and its pressing trigger).
