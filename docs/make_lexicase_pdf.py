"""Erzeugt eine 5-seitige PDF-Praesentation zu (Epsilon-)Lexicase Selection.

Stil und Hilfsfunktionen bewusst analog zu make_neat_pdf.py gehalten.
Ausgabe: docs/lexicase_selection.pdf
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyArrowPatch, Ellipse, FancyBboxPatch, Rectangle

INK    = "#1a202c"
MUTED  = "#718096"
BLUE   = "#2b6cb0"
GREEN  = "#2f855a"
ORANGE = "#c05621"
RED    = "#9b2c2c"
LIGHT  = "#edf2f7"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
    "text.color": INK,
})

ASPECT = 13.33 / 7.5


def new_slide(title, subtitle=None):
    fig = plt.figure(figsize=(13.33, 7.5))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    ax.text(6, 90, title, fontsize=27, fontweight="bold", va="top")
    if subtitle:
        ax.text(6, 83.5, subtitle, fontsize=14, color=MUTED, va="top")
    ax.plot([6, 94], [80.5, 80.5], color=LIGHT, lw=2.5, solid_capstyle="round")
    return fig, ax


def node(ax, x, y, label, color, r=2.6, fs=11, textcolor="white"):
    ax.add_patch(Ellipse((x, y), 2 * r, 2 * r * ASPECT, facecolor=color,
                         edgecolor="white", lw=1.6, zorder=3))
    ax.text(x, y, label, ha="center", va="center", fontsize=fs, color=textcolor,
            fontweight="bold", zorder=4)


def arrow(ax, p1, p2, color=MUTED, lw=1.6, style="-|>", ls="-", shrink=13, alpha=1.0):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle=style, color=color, lw=lw,
                                 linestyle=ls, alpha=alpha,
                                 shrinkA=shrink, shrinkB=shrink,
                                 mutation_scale=13, zorder=2))


def box(ax, x, y, w, h, facecolor=LIGHT, edgecolor="none", alpha=1.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.8,rounding_size=1.5",
                                facecolor=facecolor, edgecolor=edgecolor, alpha=alpha, zorder=0))


def note(ax, x, y, text, color=INK, fs=12.5, weight="normal", va="top", ha="left"):
    ax.text(x, y, text, fontsize=fs, color=color, va=va, ha=ha, fontweight=weight,
            linespacing=1.65, zorder=5)


HERE = os.path.dirname(os.path.abspath(__file__))
out = os.path.join(HERE, "lexicase_selection.pdf")
pdf = PdfPages(out)

_PNG_DIR = os.environ.get("LEXICASE_PNG_DIR")
_slide_no = [0]
_orig_savefig = pdf.savefig
def _savefig(fig):
    _orig_savefig(fig)
    _slide_no[0] += 1
    if _PNG_DIR:
        fig.savefig(os.path.join(_PNG_DIR, f"slide{_slide_no[0]}.png"), dpi=80)
pdf.savefig = _savefig

# ── Slide 1 ────────────────────────────────────────────────────────────────────
fig, ax = new_slide("Das Problem: ein Skalar verdeckt alles",
                    "Aggregierte Fitness mittelt Teilleistungen zu einer Zahl — Spezialisten verlieren gegen den soliden Durchschnitt")

heads = ["Balance", "Sinkflug", "x-Praez.", "Treibstoff", "Reward"]
x0, dx = 20, 13

note(ax, 8, 71, "Individuum S (Spezialist)", fs=12.5, weight="bold")
vals_s = [10, 9, 9, 2, 3]
note(ax, 8, 55, "Individuum D (Durchschnitt)", fs=12.5, weight="bold")
vals_d = [6, 6, 6, 6, 6]

for k, h in enumerate(heads):
    x = x0 + k * dx
    note(ax, x, 76, h, fs=10, color=MUTED, ha="center")
    node(ax, x, 66, str(vals_s[k]), BLUE if vals_s[k] >= 8 else LIGHT,
         r=2.9, fs=12, textcolor="white" if vals_s[k] >= 8 else MUTED)
    node(ax, x, 50, str(vals_d[k]), LIGHT, r=2.9, fs=12, textcolor=MUTED)

note(ax, 82, 66, "Mittel 6.6", fs=12, weight="bold", color=MUTED, va="center")
note(ax, 82, 50, "Mittel 6.0", fs=12, weight="bold", color=MUTED, va="center")

box(ax, 8, 20, 84, 20, LIGHT)
note(ax, 11, 36,
     "S beherrscht Balance und Prazision nahezu perfekt — genau die Nische, aus der eine gute Losung\n"
     "wachsen konnte. Bei aggregierter Fitness (Turnier) entscheidet aber der Mittelwert: 6.6 gegen 6.0\n"
     "ist ein Wimpernschlag, und schon eine kleine Skalen-Anderung bei \"Treibstoff\" dreht das Ergebnis.", fs=12)
note(ax, 11, 24.5,
     "Folge: die Population konvergiert auf Generalisten, seltene Teilkompetenzen sterben aus,\n"
     "Diversitat geht verloren.", fs=12, color=MUTED)
pdf.savefig(fig); plt.close(fig)

# ── Slide 2 ────────────────────────────────────────────────────────────────────
fig, ax = new_slide("Die Idee: Lexicase Selection",
                    "Spector 2012 — ein Elternteil pro Selektionsereignis, gefiltert Testfall fur Testfall in zufalliger Reihenfolge")

note(ax, 8, 73, "Ein Selektionsereignis (= ein Elternteil ziehen)", fs=13, weight="bold")

steps = [
    ("1", "ganze Population\nals Kandidatenpool", BLUE),
    ("2", "Testfalle mischen\n(neu pro Ziehung)", BLUE),
    ("3", "nachster Testfall:\nnur die Besten behalten", ORANGE),
    ("4", "Pool > 1?  weiter\nzum nachsten Testfall", ORANGE),
    ("5", "1 ubrig = Elternteil\n(sonst Zufall)", GREEN),
]
for i, (n, cap, col) in enumerate(steps):
    x = 13 + i * 17.5
    node(ax, x, 60, n, col, r=3.2, fs=14)
    note(ax, x, 53, cap, fs=10, color=MUTED, ha="center")
    if i < len(steps) - 1:
        arrow(ax, (x, 60), (x + 17.5, 60), color=MUTED, lw=1.8, shrink=34)

note(ax, 8, 40, "Warum \"lexicase\"?", fs=13, weight="bold")
note(ax, 10, 34.5,
     "Wie bei der alphabetischen (lexikografischen) Ordnung: der erste Testfall entscheidet zuerst,\n"
     "der zweite nur noch die Gleichstande des ersten, usw. Weil die Reihenfolge jedes Mal neu\n"
     "gewurfelt wird, bekommt uber viele Ziehungen jeder Testfall die Chance, \"zuerst\" zu sein.", fs=12)
note(ax, 8, 18,
     "Kein Mittelwert, keine Gewichte, keine Skalierung der Ziele. Wer irgendeinen Testfall am besten\n"
     "lost, wird mit einiger Wahrscheinlichkeit als Elternteil gewahlt — auch als reiner Spezialist.", fs=12, color=MUTED)
pdf.savefig(fig); plt.close(fig)

# ── Slide 3 ────────────────────────────────────────────────────────────────────
fig, ax = new_slide("Epsilon-Lexicase fur kontinuierliche Fitness",
                    "La Cava et al. 2016 — bei reellwertigem Reward ist \"exakt gleich gut\" nie erfullt; eine Toleranz macht das Filtern sinnvoll")

note(ax, 8, 74, "Ein Testfall, sechs Kandidaten (hoher = besser)", fs=12.5, weight="bold")

# candidate values on one test case
cand = [-30, -95, -34, -210, -31, -140]
eps  = 18                     # illustrativ vergrossert fur sichtbaren Abstand
vmin, vmax = min(cand), max(cand)          # -210 , -30
best = vmax
BASE, TOP = 34, 66                          # bar area in y
def barh(v):
    return BASE + (v - vmin) / (vmax - vmin) * (TOP - BASE)
xs = [16 + i * 12.5 for i in range(len(cand))]

yb = barh(best)
ye = barh(best - eps)
ax.plot([10, 90], [yb, yb], color=BLUE, lw=1.4, ls="--", zorder=2)
note(ax, 90.5, yb + 1.6, "best", fs=10, color=BLUE, ha="left")
ax.plot([10, 90], [ye, ye], color=ORANGE, lw=1.4, ls="--", zorder=2)
note(ax, 90.5, ye - 1.6, "best − epsilon", fs=10, color=ORANGE, ha="left", va="top")

for i, v in enumerate(cand):
    keep = v >= best - eps
    top = barh(v)
    col = GREEN if keep else LIGHT
    ax.add_patch(Rectangle((xs[i] - 3.5, BASE), 7, top - BASE, facecolor=col,
                           edgecolor="white" if keep else MUTED, lw=1.2, zorder=3))
    note(ax, xs[i], top + 2.4, str(v), fs=10,
         color=GREEN if keep else MUTED, ha="center",
         weight="bold" if keep else "normal")
    note(ax, xs[i], BASE - 2, "behalten" if keep else "raus", fs=9.5,
         color=GREEN if keep else MUTED, ha="center")

box(ax, 8, 6, 84, 20, LIGHT)
note(ax, 11, 22,
     "epsilon automatisch pro Testfall und Runde (Default, epsilon < 0):", fs=12, weight="bold")
note(ax, 11, 17.5,
     "epsilon = Median der absoluten Abweichungen vom Median (MAD) uber die aktiven Kandidaten —\n"
     "robust gegen Ausreisser, skaleninvariant, kein Handtuning. Alternativ fester Wert (epsilon >= 0).\n"
     "Filterregel:  behalte c  <=>  wert(c) >= best - epsilon.  Pool schrumpft monoton bis 1 ubrig ist.", fs=11)
pdf.savefig(fig); plt.close(fig)

# ── Slide 4 ────────────────────────────────────────────────────────────────────
fig, ax = new_slide("Umsetzung im Projekt",
                    "Population::lexicaseSelection(int E, float epsilon = -1.0f, std::string type = \"objectives\")")

box(ax, 7, 47, 41, 27, LIGHT)
note(ax, 10, 71, "type = \"objectives\"  (Default)", fs=12.5, weight="bold", color=BLUE)
note(ax, 10, 65.5,
     "5 isolierte Ziele aus network.lexicaseObjectives\n"
     "(uber alle Seeds gemittelt, hoher = besser):\n"
     "1  Balance  sum(-(|angle|+|angVel|))\n"
     "2  Sinkflug-Sicherheit  -(|vx|+|vy|)_end\n"
     "3  Horizontale Prazision  -|x|_end\n"
     "4  Treibstoff-Effizienz  sum(fuel penalty)\n"
     "5  Gym-Standard-Reward (holistisch)", fs=10.5)

box(ax, 52, 47, 41, 27, LIGHT)
note(ax, 55, 71, "type = \"seeds\"", fs=12.5, weight="bold", color=GREEN)
note(ax, 55, 65.5,
     "Testfall = ein Seed aus gymnasiumMultiSeed().\n"
     "Pro Seed ein einziges Kriterium:\n"
     "fitnessValues[seedIdx] (Rohreward)\n"
     "maximieren, MAD- oder fester epsilon.\n\n"
     "So im Beispiel lunarlander.py:\n"
     "pop.lexicaseSelection(E=1, type=\"seeds\")", fs=10.5)

note(ax, 8, 40, "Gemeinsame Mechanik", fs=13, weight="bold")
note(ax, 10, 34.5,
     "- Frisch gewurfelte Testfall-Reihenfolge pro Elternteil (std::shuffle mit *generator).\n"
     "- Filtern bis Pool <= 1; Rest per uniform random tie-break.\n"
     "- ni - E Kinder werden so gezogen; die E Elite-Individuen bleiben unverandert\n"
     "  (setElite() nach aggregierter fitness, wie in tournamentSelection()).\n"
     "- bestFit / meanFitness / minFitness werden identisch zur Turnier-Selektion gefuhrt.", fs=11.5)
note(ax, 8, 13,
     "Vorbedingung: lexicaseObjectives (5-D) bzw. fitnessValues gleich gross und nicht leer —\n"
     "befullt von gymnasiumMultiSeed() / gymnasium() via Network::fitGymnasium().", fs=11, color=MUTED)
pdf.savefig(fig); plt.close(fig)

# ── Slide 5 ────────────────────────────────────────────────────────────────────
fig, ax = new_slide("Eigenschaften, Kosten, Einordnung",
                    "Wann Lexicase hilft — und was es kostet")

note(ax, 8, 73, "Starken", fs=13, weight="bold", color=GREEN)
note(ax, 10, 68,
     "+  Erhalt Spezialisten und damit Populations-Diversitat (Druck auf \"corner cases\").\n"
     "+  Implizit mehrzielig: keine Gewichte, keine Normierung, keine Pareto-Front noting.\n"
     "+  Gut bei modularen / multimodalen Problemen (Spector: \"problem modality\").\n"
     "+  epsilon-Variante macht es auf reellwertige Rewards (Regression, RL-Return) anwendbar.", fs=11.5)

note(ax, 8, 47, "Kosten & Grenzen", fs=13, weight="bold", color=ORANGE)
note(ax, 10, 42,
     "-  Aufwand ~ O(#Eltern x #Testfalle x Poolgrosse) pro Generation.\n"
     "-  Braucht aussagekraftige, moglichst unkorrelierte Testfalle — hier nur 5 Ziele bzw. #Seeds.\n"
     "-  Bei sehr vielen quasi-identischen Testfallen degeneriert es Richtung Zufallswahl.", fs=11.5)

box(ax, 8, 12, 84, 16, LIGHT)
note(ax, 11, 24.5, "Einordnung im Projekt", fs=12, weight="bold")
note(ax, 11, 20,
     "tournamentSelection() = ein Skalar, paretoTournamentSelection() = Dominanz auf Ziel-Vektor,\n"
     "lexicaseSelection() = Filtern Testfall fur Testfall. Selbe Schnittstelle (Elite E, Bookkeeping),\n"
     "untereinander austauschbar in der Evolutionsschleife.", fs=11)

note(ax, 92, 4,
     "Spector (2012); Helmuth, Spector & Matheson (2015); La Cava, Spector & Danai (2016)",
     fs=9.5, color=MUTED, ha="right")
pdf.savefig(fig); plt.close(fig)

pdf.close()
print("geschrieben:", out)
