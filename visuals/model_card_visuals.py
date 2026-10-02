"""Standalone charts and optional one-page model card for the saved experiment.

Install: pip install numpy matplotlib
Run charts: python model_card_visuals.py
For the card: pip install reportlab pymupdf
Then run: python model_card_visuals.py --card
Outputs are placed beside this script. No project dataset or model is required.
The embedded values describe the delivered experiment, not a new evaluation.
"""
from pathlib import Path
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent
TEAL, BLUE, RED = '#16756d', '#477caf', '#ba4054'
# Source: outputs/classification_report.csv (saved held-out test results).
CLASSES = ['Credit reporting', 'Debt collection', 'Bank accounts', 'Credit cards',
           'Mortgages', 'Money services', 'Vehicle loans', 'Student loans',
           'Payday/title/personal loans']
SUPPORT = np.array([540, 74, 45, 44, 15, 14, 10, 9, 6])
F1 = np.array([.9460201280878316, .7464788732394366, .7578947368421053,
               .7526881720430108, .8888888888888888, .72,
               .5882352941176471, .75, 0.0])
# Source: outputs/audit.csv. F1 uses the labels present in each subgroup.
SHORT = np.array([.8389830508474576, .5986519607843137, .819574609504819])
LONG = np.array([.8897338403041825, .6843607359181436, .8874112057838611])


def setup():
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42, 'ps.fonttype': 42,
                         'axes.labelcolor': '#253238', 'text.color': '#253238'})


def save(fig, stem):
    for ext in ('png', 'pdf'):
        fig.savefig(OUT / f'{stem}.{ext}', dpi=300, facecolor='white')
    plt.close(fig)


def per_class_chart():
    """Horizontal paired bars with separate count and F1 axes; never stack units."""
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(10.6, 4.5), sharey=True,
                                  gridspec_kw={'width_ratios': [1, 1.12]})
    fig.subplots_adjust(left=.255, right=.965, bottom=.14, top=.81, wspace=.20)
    y = np.arange(len(CLASSES))
    ax.barh(y, SUPPORT, color=BLUE, height=.66)
    bx.barh(y, F1, color=[TEAL]*8+[RED], height=.66)
    bx.plot(0, 8, 'o', color=RED, markersize=5, clip_on=False)
    ax.set_yticks(y, CLASSES)
    ax.invert_yaxis()
    ax.set_xlim(0, 610)
    bx.set_xlim(0, 1.10)
    bx.set_xticks([0, .25, .5, .75, 1])
    ax.set_xlabel('Test support (n)')
    bx.set_xlabel('Per-class F1 (0-1)')
    for i, (n, f) in enumerate(zip(SUPPORT, F1)):
        ax.text(n+8, i, str(n), va='center', fontsize=9)
        bx.text(f+.025, i, f'{f:.3f}', va='center', fontsize=9,
                color=RED if f == 0 else '#253238')
    for a in (ax, bx):
        a.grid(axis='x', color='#e2e7e6', linewidth=.6)
        a.set_axisbelow(True)
        a.tick_params(axis='y', length=0)
    fig.text(.02, .955, 'Class imbalance hides weak minority-category performance', fontsize=13, weight='bold')
    fig.text(.02, .89, 'Weighted-F1 0.8842  |  Macro-F1 0.6834  |  Credit reporting: 540/757 cases (71.33%)', fontsize=10)
    save(fig, 'Per_Class_Performance')


def subgroup_chart():
    """Grouped bars on a common 0-1 scale; accuracy has saved bootstrap CIs."""
    fig, ax = plt.subplots(figsize=(10.6, 3.2))
    fig.subplots_adjust(left=.08, right=.98, bottom=.23, top=.76)
    x = np.arange(3)
    width = .30
    for offset, vals, color, label in [(-width/2, SHORT, BLUE, 'Short: <=50 words (n=118)'),
                                       (width/2, LONG, TEAL, 'Long: 200+ words (n=263)')]:
        bars = ax.bar(x+offset, vals, width, color=color, label=label)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x()+width/2, val+.065, f'{val:.3f}', ha='center', fontsize=10)
    for pos, value, lo, hi in [(-width/2, SHORT[0], .7711864406779662, .8983050847457628),
                               (width/2, LONG[0], .8517110266159695, .9239543726235742)]:
        ax.errorbar(pos, value, yerr=[[value-lo], [hi-value]], fmt='none',
                    color='#253238', capsize=4, linewidth=1)
    ax.set_xticks(x, ['Accuracy', 'Macro-F1', 'Weighted-F1'])
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0, .25, .5, .75, 1])
    ax.set_ylabel('Score (0-1)')
    ax.grid(axis='y', color='#e2e7e6', linewidth=.6)
    ax.set_axisbelow(True)
    fig.text(.02, .98, 'Short complaints have lower observed accuracy', fontsize=13, weight='bold', va='top')
    fig.legend(loc='upper center', bbox_to_anchor=(.56, .90), ncol=2, frameon=False, fontsize=10)
    fig.text(.02, .035, 'Accuracy whiskers: 95% bootstrap intervals. Length is not formality; medium texts omitted from this comparison.', fontsize=9)
    save(fig, 'Subgroup_Performance')


def model_card():
    from reportlab.pdfgen import canvas
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.platypus import Paragraph
    from reportlab.lib.colors import HexColor
    import fitz
    path = OUT / 'Model_Card.pdf'
    c = canvas.Canvas(str(path), pagesize=(595.28, 841.89))
    c.setTitle('Automated Financial Customer Complaint Classification - Model Card v2')
    style = ParagraphStyle('body', fontName='Helvetica', fontSize=8.1, leading=10.4, textColor=HexColor('#253238'))

    def text(txt, x, top, width=523, heading=False):
        s = ParagraphStyle('heading', parent=style, fontName='Helvetica-Bold', fontSize=9.4, leading=12, textColor=HexColor(TEAL)) if heading else style
        para = Paragraph(txt, s)
        _, height = para.wrap(width, 800)
        para.drawOn(c, x, 841.89-top-height)
        return top+height

    text('MODEL CARD / VERSION 2.0 / ACADEMIC PROTOTYPE / 20 SEPTEMBER 2026', 36, 24)
    c.setFillColor(HexColor('#163e3c'))
    c.setFont('Helvetica-Bold', 19)
    c.drawString(36, 789, 'Automated Financial Customer')
    c.drawString(36, 767, 'Complaint Classification')
    text('<b>TF-IDF + balanced Linear SVM (C=1)</b> | Nine product categories | Human-reviewed routing only', 36, 82)
    text('<b>TEST ACCURACY 88.90%</b> &nbsp; | &nbsp; <b>MACRO-F1 0.6834</b> &nbsp; | &nbsp; <b>WEIGHTED-F1 0.8842</b>', 36, 102)
    text('Data and scope',36,122,heading=True)
    text('CFPB: first 10,000 API records, retrieved 13 Sep 2026; recorded dates 26 Aug-4 Sep 2024. Not a random database sample. Filtering retained <b>5,042</b>: <b>4,724 duplicates</b>, 190 conflicting-label records, 7 texts with fewer than 3 cleaned words, and 37 rare-label records removed (22 prepaid cards; 15 debt/credit management). <b>Empty/missing fields removed: 0</b>; the API request already required narratives.',36,137)
    text('Training and intended use',36,194,heading=True)
    text('English-text routing suggestions, not decisions on eligibility, redress or complaint validity. Cleaned word unigrams/bigrams; TF-IDF fitted on training only (max 30,000 features). Stratified train/validation/test: 3,529/756/757, seed 42. Selected by validation macro-F1 against logistic regression, random forest and baseline. Test accuracy 95% bootstrap interval: 86.66%-91.02%.',36,209)
    c.drawImage(str(OUT/'Per_Class_Performance.png'), 30, 841.89-473, width=535, height=227.1)
    text('Weighted-F1 weights classes by support; Macro-F1 gives each class equal weight. The dominant class scores well, while payday/title/personal loans has F1=0 and recall=0 (n=6).',36,472)
    c.drawImage(str(OUT/'Subgroup_Performance.png'), 30, 841.89-660, width=535, height=161.5)
    text('Length gap: long minus short = <b>5.08 percentage points</b>; not a causal finding. Formal-detailed proxy: 85.84% (n=113); informal-short: 80.00% (n=5). Style comparison is inconclusive. Human-rated formality and demographic fairness have not been established.',36,659)
    text('Outputs, explainability and human triage',36,695,heading=True)
    text('Output: highest-scoring category. Raw SVM scores/margins are <b>uncalibrated, not probabilities</b>. Explanations use feature weights and TF-IDF x weight contributions (not SHAP). <b>Current code:</b> rejects &lt;3 cleaned words or no known vocabulary; otherwise predicts without a margin threshold. <b>Proposed, not implemented/validated:</b> general human queue for invalid input, top score &lt;=0.10, or top-minus-runner-up gap &lt;=0.10; otherwise category-specific human review. These illustrative cutoffs require validation-set tuning; no autonomous routing is approved.',36,710)
    text('<b>Governance:</b> Keep human oversight and narrative privacy controls; validate broader/temporal samples, annotate style, and monitor class/subgroup errors before deployment. Thresholds are documentation proposals, not measured safeguards.',36,777)
    text('Evidence: outputs/{classification_report,audit}.csv; test_metrics.json; data_manifest.json; data/provenance.json; src/project.py. Filtering counts recomputed from the delivered CSV. Charts use saved test results; no retraining.',36,809)
    c.save()
    doc=fitz.open(path)
    assert len(doc)==1
    doc[0].get_pixmap(matrix=fitz.Matrix(300/72,300/72)).save(OUT/'Model_Card.png')


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--card', action='store_true', help='Also generate the one-page PDF and PNG card.')
    args=parser.parse_args()
    assert SUPPORT.sum()==757
    assert np.isclose(F1.mean(), .68335623257988)
    assert np.isclose(np.average(F1, weights=SUPPORT), .8842265981366503)
    setup()
    per_class_chart()
    subgroup_chart()
    if args.card:
        model_card()
    print(f'Created assets in {OUT}')
