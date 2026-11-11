# 🏈 Does the NFL Combine Predict NFL Success?

![Status](https://img.shields.io/badge/status-analysis%20done-brightgreen)
![Tests](https://img.shields.io/badge/tests-258%20passing-brightgreen)
![Players](https://img.shields.io/badge/players-~400%20(2023--2025%20draftees)-blue)
![Tests run](https://img.shields.io/badge/comparisons-3%2C709-orange)
![Competition](https://img.shields.io/badge/Kaggle-NFL%20Big%20Data%20Bowl%202027-20BEFF)

> **The question:** If a player runs fast, jumps high or moves well at the Combine, does he play better in the NFL? And which traits are actually worth a college program's training time?

We linked **10 Hz Combine sensor tracking** to **NFL game tracking** for players drafted in 2023–2025 and checked, by position group, what carries over.

---

## 🧭 The short answer

| | |
|---|---|
| 🟢 **What holds up** | **Heavier receivers get less separation from defenders** (about 0.023 yards less per 5 lb). This is the one result that survives our strictest correction. |
| 🟡 **Leads worth watching** | Pass rushers who accelerate and burst hard get more pressures. Faster offensive linemen start more games. The NFL's college-production score predicts how often receivers are targeted. |
| 🔴 **What did not hold up** | The NFL's own athleticism score predicts **nothing** about game performance. Most speed and agility drills do not translate. |

> [!IMPORTANT]
> This is **association, not causation**. Better players may simply test well. Nothing here proves that training a trait makes a player better.

---

## 🗺️ What we did, step by step

1. **📥 Built the data.** Joined Combine results, Combine tracking and three seasons of game tracking for DL, OL, WR, TE and DB.
2. **📏 Measured how players actually play.** For example: how fast a pass rusher gets off the ball, how quickly a lineman reacts at the snap, how much separation a receiver gets.
3. **✅ Checked the measures are real.** A game measure only counts if a player scores about the same in random halves of his own plays.
4. **🧪 Ran one planned test.** Written down before looking: do sensor measures add anything beyond the usual Combine numbers?
5. **🔍 Searched everything else.** 3,709 comparisons across all five position groups, every drill, and the NFL's own scores. Every plan was logged before results, and all tests were pooled into one correction so lucky hits are not mistaken for findings.
6. **🧑‍🏫 Translated it for coaches.** Each trait tagged trainable or fixed, with effects in coach units ("per 0.1 s", "per 2 inches"), power checks and evidence grades.

---

## 📊 Results by position group

| Group | Players | What we found |
|---|---|---|
| 🛡️ **Defensive line** | ~100 | Hard acceleration and burst in the 40 go with more hurries, pressures and tackles for loss (a lead). Heavier DL get off the ball slower. |
| 🧱 **Offensive line** | ~100 | Nothing in the planned test. Faster OL tend to start more games (a lead). |
| 🎯 **Wide receivers / TE** | ~100 | Heavier receivers get less separation (**confirmed**). College production predicts targets. Speed and route-drill times do not. |
| 🧤 **Defensive backs** | ~120 | Nothing survives. Several drills line up with run-pursuit speed (leads only). |

> [!NOTE]
> The Combine tracking data only covers **DB, DL, OL, WR and TE**. There are no running backs, quarterbacks or linebackers.

---

## 🧾 How strong is each finding?

| Grade | Meaning | Count |
|---|---|---|
| 🟢 **Supported** | Survives the correction across all 3,709 tests | **1** finding (2 rows) |
| 🟡 **Suggestive** | Looks real, but does not survive the full correction | **~8** |
| 🔴 **Speculative** | Could easily be luck | Everything else |

See the full ranked list in [`findings.md`](findings.md#correlation-shortlist-fixed-rule-no-new-tests).

---

## ⚠️ Why so many nulls?

- 🔢 Only **~100 players per position**, which can catch large effects but not small ones.
- 🏟️ Game results depend on scheme, teammates and the quarterback, not just the player.
- 🔁 Many drills were run **once per player**, so we cannot tell skill from luck.
- 📍 Some effects are really a **position** effect (edge rushers are both faster and rush more).

---

## 📁 What is in this repo

| File | What it is |
|---|---|
| [`findings.md`](findings.md) | The full research summary, written in plain English |
| [`COACHING.md`](COACHING.md) | Coach-facing tables: prioritize, deprioritize or unclear, per position |
| [`DL.md`](DL.md) / [`OL.md`](OL.md) | Detailed defensive line and offensive line write-ups |
| `src/combine_dlol/` | The analysis code |
| `tests/` | 258 automated tests |
| `scripts/exploratory/` | Scan and correction scripts |

> Raw data and result tables are kept local and are not on GitHub.

---

## ✅ Done and 🔜 next

- [x] Data pipeline and game-side measures
- [x] Planned test and all exploratory scans
- [x] Coaching-unit effects, power checks and evidence grades
- [x] Findings and coaching write-ups
- [ ] Public Kaggle notebook (required by the competition)
- [ ] Figures
- [ ] Final writeup
- [ ] Join with the cut-score work

---

<sub>Built for the Kaggle NFL Big Data Bowl 2027. Deadline: January 6, 2027.</sub>
