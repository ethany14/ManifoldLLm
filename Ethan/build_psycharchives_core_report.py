"""Build the HOW-focused supervisor report from verified aggregate results.

No licensed person-level files are opened by this builder.
"""

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    HRFlowable, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
)

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output" / "pdf" / "PSYCHARCHIVES_CORE_METHOD_REPORT_EN.pdf"
NAVY = colors.HexColor("#17324d")
TEAL = colors.HexColor("#176b77")
INK = colors.HexColor("#233747")
MUTED = colors.HexColor("#596c7b")
PALE = colors.HexColor("#edf4f6")
LIGHT = colors.HexColor("#f7f9fa")
RULE = colors.HexColor("#d5e0e5")


def setup():
    fonts = Path("C:/Windows/Fonts")
    pdfmetrics.registerFont(TTFont("Report", str(fonts / "arial.ttf")))
    pdfmetrics.registerFont(TTFont("Report-Bold", str(fonts / "arialbd.ttf")))
    pdfmetrics.registerFontFamily("Report", normal="Report", bold="Report-Bold")
    return {
        "title": ParagraphStyle("title", fontName="Report-Bold", fontSize=18.5,
                                leading=23, textColor=NAVY, spaceAfter=9),
        "deck": ParagraphStyle("deck", fontName="Report", fontSize=10.2,
                               leading=15, textColor=MUTED, spaceAfter=13),
        "h1": ParagraphStyle("h1", fontName="Report-Bold", fontSize=12.1,
                             leading=17, textColor=NAVY, spaceBefore=11,
                             spaceAfter=6, keepWithNext=True),
        "h2": ParagraphStyle("h2", fontName="Report-Bold", fontSize=9.8,
                             leading=14, textColor=TEAL, spaceBefore=8,
                             spaceAfter=4, keepWithNext=True),
        "body": ParagraphStyle("body", fontName="Report", fontSize=9,
                               leading=13.7, textColor=INK, spaceAfter=7),
        "small": ParagraphStyle("small", fontName="Report", fontSize=8,
                                leading=11.8, textColor=MUTED, spaceAfter=5),
        "call": ParagraphStyle("call", fontName="Report-Bold", fontSize=9.2,
                               leading=14, textColor=NAVY),
        "cell": ParagraphStyle("cell", fontName="Report", fontSize=7.7,
                               leading=10.6, textColor=INK),
        "head": ParagraphStyle("head", fontName="Report-Bold", fontSize=7.7,
                               leading=10.7, textColor=colors.white),
        "ref": ParagraphStyle("ref", fontName="Report", fontSize=7.9,
                              leading=11.4, textColor=INK, leftIndent=16,
                              firstLineIndent=-16, spaceAfter=5),
    }


def p(text, style):
    return Paragraph(text, style)


def table(rows, widths, styles, header=True):
    data = [
        [p(str(value), styles["head"] if header and i == 0 else styles["cell"])
         for value in row]
        for i, row in enumerate(rows)
    ]
    element = Table(data, colWidths=widths, repeatRows=1 if header else 0,
                    hAlign="LEFT")
    spec = [
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LINEBELOW", (0, -1), (-1, -1), 0.5, RULE),
    ]
    if header:
        spec += [
            ("BACKGROUND", (0, 0), (-1, 0), NAVY),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
        ]
    else:
        spec += [("BACKGROUND", (0, 0), (-1, -1), PALE)]
    element.setStyle(TableStyle(spec))
    return element


def footer(canvas, doc):
    canvas.saveState()
    w, _ = A4
    canvas.setStrokeColor(RULE)
    canvas.line(53, 45, w - 53, 45)
    canvas.setFont("Report", 7.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(53, 32, "How to use manifold geometry for digital personality | interim method report")
    canvas.drawRightString(w - 53, 32, str(doc.page))
    canvas.restoreState()


def build():
    s = setup()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(OUT), pagesize=A4, leftMargin=53, rightMargin=53,
        topMargin=47, bottomMargin=61,
        title="How to Use a Manifold to Form a Digital Personality",
        author="Digital Personality Manifold research team",
    )
    story = []
    add = story.append

    # Page 1: answer and conceptual architecture.
    add(p("How to Use a Manifold to Form a Digital Personality", s["title"]))
    add(p("A literature-derived implementation path and what the current longitudinal experiment establishes | 20 September 2026", s["deck"]))
    add(table([[
        "Direct answer",
        "Use a learned slow person coordinate as a chart of individual tendencies; let a nonlinear decoder turn movement in that chart into changes in observable affect and behavior; derive local distance from the decoder; train and update the person estimate from ordered observations. The resulting manifold is an operational model, not a picture or an emotion label."
    ]], [105, 383], s, header=False))
    add(Spacer(1, 9))
    add(p("1. What is the manifold here?", s["h1"]))
    add(p("For person i at observation t, an ordered history H<sub>i,<=t</sub> produces a slow posterior q(phi<sub>it</sub> | H<sub>i,<=t</sub>). The current cues and slow state produce a fast posterior q(z<sub>it</sub> | x<sub>it</sub>, phi<sub>it</sub>). A nonlinear decoder g(phi, z) predicts a vector of observable cues. The <b>person manifold</b> is the coordinate domain of phi <i>equipped with a decoder-induced metric</i>, so equal coordinate steps need not imply equal observable change [4,5]. It is not an Isomap embedding and is not the raw LLM hidden space.", s["body"]))
    add(p("The distinction matters: a neural network inevitably uses coordinates, but a <b>nonlinear manifold method</b> adds a local metric and geometry-aware operation to those coordinates. A plain autoencoder or latent vector without those operations is only a comparison model. The current decoder metric is calculated on the 4-D slow space while fast state is held at a reference value; psychological validity is a separate empirical claim.", s["body"]))
    add(p("Figure 1. Proposed person-state pipeline and the role of geometry", s["h2"]))
    add(table([
        ["Observed history", "Person and state inference", "Geometry", "Use and update"],
        ["Repeated affect, sensing, context, time order", "Slow q(phi | history) plus fast q(z | current state)", "Decoder g(phi,z) defines local G(phi) and path length", "Same-person learning, retrieval, future behavior; update after new records"],
    ], [110, 132, 128, 118], s))
    add(Spacer(1, 7))
    add(p("<b>Research choice.</b> We recommend a temporal slow-fast generative model with a decoder-induced Riemannian metric as the <i>method to investigate</i>, because it makes person dynamics and manifold distance explicit in one learnable system. This is a design synthesis, not a literature-proven best model or a claim that the current geometry improves prediction.", s["body"]))
    add(p("The literature provides different pieces rather than one complete solution: Whole Trait Theory and PersDyn motivate stable tendencies alongside state variation [1,2]; neural personality-process models motivate explicit person-situation computations [3]; variational autoencoding motivates revisable uncertain latent inference [4]; decoder geometry supplies the metric [5,6]. We combine these pieces in a new longitudinal person-level application.", s["small"]))

    # Page 2: implementable algorithm.
    add(PageBreak())
    add(p("2. A concrete training and inference recipe", s["h1"]))
    add(p("Step A - collect and align repeated observations", s["h2"]))
    add(p("A training record must link many time-ordered observations to the same consenting person. Each record should contain available affect, behavior, and context; record missingness rather than silently treating it as zero. Person-level questionnaires can anchor interpretation, while later behavior supplies an independent outcome. Isolated emotion sentences cannot identify a stable person coordinate. The current PsychArchives panel provides repeated affect, sensing, and person-level traits, but not participant-linked language [8].", s["body"]))
    add(p("Step B - infer slow and fast variables", s["h2"]))
    add(p("A causal, missingness-aware GRU reads x<sub>i,1:t</sub> and emits q(phi<sub>it</sub>), a four-dimensional slow Gaussian posterior. The current observation and phi produce q(z<sub>it</sub>), a four-dimensional fast posterior. A transition predicts z<sub>i,t+1</sub> and the next-questionnaire valence. Slow-state regularization favors gradual updates; the fast state may react to immediate context. These are modeling choices inspired by dynamic trait/state theories [1-3], not a direct implementation of PersDyn attractor strength.", s["body"]))
    add(p("Step C - learn the decoder and its local metric", s["h2"]))
    add(p("Train a nonlinear multidimensional decoder g(phi,z) to reconstruct observed cue dimensions. At a prespecified reference fast state z=0, calculate", s["body"]))
    add(table([["Local geometry", "G(phi) = J<sub>phi</sub>g(phi,0)<super>T</super> J<sub>phi</sub>g(phi,0) + 10<super>-4</super>I"]], [106, 382], s, header=False))
    add(Spacer(1, 7))
    add(p("J is the decoder Jacobian. The metric measures how much a small move in person coordinates changes the decoded observation vector. Our formula is a <b>simplified deterministic pullback</b> motivated by Arvanitidis et al. [5]; it is not their full stochastic decoder metric or the Fisher-Rao construction of [6]. The ridge stabilizes numerics and can make eigenvalues positive even if the data-informed Jacobian lacks full rank. Check rank, conditioning, density coverage, and sensitivity to the reference state before interpreting distances.", s["body"]))
    add(p("Step D - make geometry do work during training", s["h2"]))
    add(p("Form nonoverlapping early and late windows from each training person. Add an identity contrastive objective that makes matching windows close and other people farther apart. Compare d<sub>E</sub> = ||phi<sub>a</sub> - phi<sub>b</sub>|| with d<sub>path</sub>, the sum of decoded changes along a sampled straight coordinate path from phi<sub>a</sub> to phi<sub>b</sub>. Backpropagate through the decoder path. This is the project's extension of decoder geometry into a person-consistency task; the cited geometry papers did not train digital-personality identities. A sampled straight path is <b>not</b> a minimized geodesic.", s["body"]))
    add(p("The weighted objective combines next-valence error, observed-cue reconstruction, fast-transition consistency, slow-state stability, a KL penalty, optional trait loss, and optional identity contrastive loss. It is VAE-inspired [4], but the weighted multitask loss is <b>not</b> claimed to be a standard ELBO. Geometry needs a matched Euclidean-loss control; otherwise any benefit could come from identity supervision rather than curvature.", s["small"]))

    # Page 3: traceable paper comparison.
    add(PageBreak())
    add(p("3. Where each component came from - and what changed", s["h1"]))
    add(p("The table distinguishes a paper's actual contribution from our adaptation. None of the cited papers demonstrates that a manifold has already formed a valid digital personality.", s["body"]))
    add(table([
        ["Source", "Original contribution", "What we borrow or change", "Boundary"],
        ["Whole Trait Theory [1]", "Traits include distributions of momentary states and explanatory situation responses.", "Person tendency plus current state; require cross-situation evidence.", "Not a geometric learning method."],
        ["PersDyn [2]", "Baseline, variability, and attractor strength summarize state dynamics.", "Slow phi and fast z approximate the stable/dynamic split.", "We do not estimate PersDyn attractor strength or continuous-time regulation."],
        ["Neural personality process [3]", "Explicit neural computation of personality structure and within-person dynamics.", "An ordered, person-situation-state computation from observational data.", "Our GRU does not implement that paper's motive/goal architecture."],
        ["Variational autoencoder [4]", "Approximate posterior learning and generative decoding.", "Gaussian slow/fast posteriors, decoder, KL regularization.", "Our weighted temporal objective is not the paper's i.i.d. ELBO."],
        ["Latent Space Oddity [5]", "A generative decoder induces non-Euclidean latent geometry.", "Compute J<super>T</super>J on slow person coordinates and use decoded path lengths in identity training.", "We use a deterministic z=0 slice, not the full stochastic metric or exact geodesics."],
        ["Pulling back information geometry [6]", "Fisher-Rao geometry extends pullbacks to decoder distributions.", "Possible extension for probabilistic mixed cue types.", "Not implemented in the current experiment."],
        ["Isomap [7]", "Neighborhood graph estimates global geodesic structure for embedding static observations.", "Useful nonparametric diagnostic or alternative baseline.", "No Isomap target or mapping is used in this training."],
    ], [105, 137, 147, 99], s))
    add(Spacer(1, 8))
    add(p("<b>Key distinction from previous work:</b> the substantive adaptation is not merely placing an autoencoder beside a personality study. We define geometry specifically on a <i>slow, repeatedly updated person coordinate</i>, conditioned on fast state through a decoder; then put its path distance inside matched longitudinal person-consistency training. The scientific question is whether that added geometry improves an external personality-relevant outcome beyond the same model using Euclidean distance.", s["body"]))
    add(p("The papers justify components and mathematical operations, not our data-to-person interpretation. In particular, a changing positive-definite G(phi) is insufficient to establish intrinsic curvature, a unique psychological axis, or a valid digital persona.", s["small"]))

    # Page 4: evidence, subordinate to method.
    add(PageBreak())
    add(p("4. Current experiment: an engineering check on the path", s["h1"]))
    add(p("The licensed PsychArchives data have 9,790 repeated snapshots from 455 people; 380 have complete five-trait parameters [8]. Person-disjoint splits are 318 train, 68 development, and 69 reserved test. The ordered model uses 22 cues, with 6,534/1,377 train/development next-questionnaire pairs. The development cohort has been inspected repeatedly; the results below are exploratory, not confirmatory.", s["body"]))
    add(p("Does geometry add value beyond the same identity task?", s["h2"]))
    add(table([
        ["Development result; three-seed mean", "No identity loss", "Euclidean identity", "Decoder-path identity"],
        ["Next-valence RMSE (lower)", "0.873", "0.890", "0.879"],
        ["Five-trait probe macro R<super>2</super> (higher)", "0.071", "0.135", "0.138"],
        ["Disjoint-window rank-1, own distance (higher)", "11.3% Eucl.", "16.7% Eucl.", "16.7% path"],
    ], [214, 86, 94, 94], s))
    add(Spacer(1, 7))
    add(p("The corrected decoder-path loss sends gradients through the nonlinear decoder, so manifold geometry is genuinely used during training. However, both identity losses improve person matching similarly. The 0.003 trait R<super>2</super> difference does not establish a nonlinear-geometry advantage; different readouts favor different conditions. The robust observation is a benefit from <i>same-person contrastive learning</i>, not yet from Riemannian distance.", s["body"]))
    add(p("Does the person coordinate carry personality information beyond simpler cues?", s["h2"]))
    add(table([
        ["Input / summary", "Trait macro R<super>2</super>", "Rank-1 same-person match"],
        ["Slow phi, all 22 cues", "0.126", "16.7%"],
        ["Slow phi, time and valence only", "0.151", "5.9%"],
        ["Slow phi, valence only", "0.145", "6.4%"],
        ["Person valence mean + SD, post-hoc", "0.159", "7.4%"],
    ], [248, 112, 128], s))
    add(Spacer(1, 7))
    add(p("The feature-group models use zero trait loss and the same Euclidean identity objective. Full sensing cues greatly raise person matching without raising measured-trait prediction. A post-hoc two-number valence summary predicts traits better than these learned slow coordinates on the 57 complete-trait development people. Person matching may reflect habits, context, or device patterns; it cannot by itself certify personality. The simple baseline is a diagnostic added after results inspection, not a preregistered winner.", s["body"]))
    add(p("<b>Method decision from the experiment:</b> keep the temporal decoder-metric approach as an implemented research path, but do not claim it is empirically the best personality representation. Retain Euclidean and simple-summary baselines in every next test. The 69-person test split remains unscored.", s["small"]))

    # Page 5: implementation route, claims, references.
    add(PageBreak())
    add(p("5. Final method conclusion and next implementation stages", s["h1"]))
    add(p("<b>How to use manifold:</b> fit a causal slow-fast generative model to person-linked longitudinal observations; treat slow phi as a revisable coordinate chart, not a trait label; define G(phi) from a multi-output nonlinear decoder; use that geometry in a prespecified distance-based training or retrieval operation; and test the resulting behavior or trait validity against the same architecture with Euclidean distance and simple affective summaries. Store the posterior, metric diagnostics, time, evidence provenance, and uncertainty as the person's updateable digital state.", s["body"]))
    add(table([
        ["Stage", "Deliverable and decision criterion", "Status"],
        ["1. Person-state prototype", "Ordered history to q(phi), q(z), transition, decoder, local G(phi); online re-inference.", "Implemented on PsychArchives."],
        ["2. Geometry-specific test", "Prespecify reference contexts and path solver; compare path or optimized geodesic with matched Euclidean use, including rank/conditioning checks.", "Fixed-path comparison done; distinctive gain not shown."],
        ["3. Personality validation", "Predict independent later behavior across contexts; control simple mean/variability, traits, history/time, missingness, and device/site effects. Freeze analysis before untouched evaluation or use an independent cohort.", "Needed before a validated personality claim."],
        ["4. Language interface", "With consented person-linked longitudinal text, condition a language model on q(phi), z, and uncertainty; compare flat and geometry-aware conditioning on independent behavioral/contextual outcomes.", "Proposed only; no LLM persona trained here."],
    ], [96, 284, 108], s))
    add(Spacer(1, 8))
    add(p("<b>What the report can conclude now:</b> a literature-grounded computational answer to <i>how</i> is available. It is a candidate digital-personality manifold formed through a temporal posterior and a decoder pullback metric. The available experiment supports feasibility and person-specific signal, but not psychological construct validity or superiority of nonlinear geometry. If an independent study finds no geometry-specific gain, that is a valid result about this implementation rather than a failure to define a manifold.", s["body"]))
    add(p("References", s["h1"]))
    refs = [
        "[1] Fleeson, W., &amp; Jayawickreme, E. (2015). Whole Trait Theory. <i>Journal of Research in Personality, 56</i>, 82-92. <link href='https://pmc.ncbi.nlm.nih.gov/articles/PMC4472377/' color='#176b77'>Full text</link>.",
        "[2] Sosnowska, J., Kuppens, P., De Fruyt, F., &amp; Hofmans, J. (2020). New directions in the conceptualization and assessment of personality: A dynamic systems approach. <i>European Journal of Personality, 34</i>, 988-998. <link href='https://doi.org/10.1002/per.2233' color='#176b77'>DOI</link>.",
        "[3] Read, S. J., Droutman, V., Smith, B. J., &amp; Miller, L. C. (2019). Using neural networks as models of personality process: A tutorial. <i>Personality and Individual Differences, 136</i>, 52-67. <link href='https://pmc.ncbi.nlm.nih.gov/articles/PMC6411310/' color='#176b77'>Full text</link>.",
        "[4] Kingma, D. P., &amp; Welling, M. (2014). Auto-Encoding Variational Bayes. <i>ICLR.</i> <link href='https://arxiv.org/abs/1312.6114' color='#176b77'>Paper</link>.",
        "[5] Arvanitidis, G., Hansen, L. K., &amp; Hauberg, S. (2018). Latent space oddity: On the curvature of deep generative models. <i>ICLR.</i> <link href='https://arxiv.org/abs/1710.11379' color='#176b77'>Paper</link>.",
        "[6] Arvanitidis, G., et al. (2022). Pulling back information geometry. <i>AISTATS, PMLR 151</i>, 4872-4894. <link href='https://proceedings.mlr.press/v151/arvanitidis22b.html' color='#176b77'>Paper</link>.",
        "[7] Tenenbaum, J. B., de Silva, V., &amp; Langford, J. C. (2000). A global geometric framework for nonlinear dimensionality reduction. <i>Science, 290</i>, 2319-2323. <link href='https://doi.org/10.1126/science.290.5500.2319' color='#176b77'>DOI</link>.",
        "[8] Schoedel, R., et al. <i>Dataset for: Snapshots of Daily Life: Situations Investigated Through the Lens of Smartphone Sensing.</i> PsychArchives. <link href='https://psycharchives.org/en/item/975a7624-cd16-46bb-9d80-d988bed5e780' color='#176b77'>Dataset record</link>.",
    ]
    for ref in refs:
        add(p(ref, s["ref"]))
    add(HRFlowable(width="100%", thickness=0.5, color=RULE,
                       spaceBefore=6, spaceAfter=6))
    add(p("The cited papers provide theory or methods, not the project results. Project scores are from development-only aggregate runs; licensed participant-level records are not reproduced here.", s["small"]))
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(OUT)


if __name__ == "__main__":
    build()
