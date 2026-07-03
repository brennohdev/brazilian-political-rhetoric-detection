# Annotation Protocol

## Overview

This document defines the annotation guidelines for detecting rhetorical manipulation techniques in Brazilian parliamentary speech segments. Two annotators (Brenno and Gustavo) independently label each segment, then disagreements are resolved through discussion.

---

## 1. Task Definition

For each segment (3-5 sentences from a parliamentary speech), the annotator must decide:

1. **Which techniques are present?** (multi-label: zero, one, or more)
2. **What is the span?** (the exact text that contains the technique)

A segment may contain:
- No techniques (most common — normal political speech)
- One technique
- Multiple different techniques
- Multiple instances of the same technique

---

## 2. Google Sheets Structure

### Sheet Setup

Create one Google Sheets workbook with 3 tabs:

#### Tab 1: "segments" (read-only reference)

| Column | Content |
|--------|---------|
| A: segment_id | Unique ID (e.g., `proc_PT_Fulano_2023-05-10_abc123_seg001`) |
| B: text | Full segment text |
| C: deputy_name | Who spoke |
| D: party | Party abbreviation |
| E: political_spectrum | left / center / right |

#### Tab 2: "annotator_brenno" (Brenno fills this)

| Column | Content | Instructions |
|--------|---------|--------------|
| A: segment_id | Copy from tab 1 | Don't modify |
| B: text | Copy from tab 1 (for easy reading) | Don't modify |
| C: has_technique | YES / NO | Does this segment contain ANY technique? |
| D: loaded_language | 0 or 1 | Is Loaded Language present? |
| E: name_calling | 0 or 1 | Is Name Calling present? |
| F: doubt | 0 or 1 | Is Doubt present? |
| G: appeal_to_fear | 0 or 1 | Is Appeal to Fear present? |
| H: causal_oversimplification | 0 or 1 | Is Causal Oversimplification present? |
| I: flag_waving | 0 or 1 | Is Flag-Waving present? |
| J: span_loaded_language | text | Exact span (if D=1) |
| K: span_name_calling | text | Exact span (if E=1) |
| L: span_doubt | text | Exact span (if F=1) |
| M: span_appeal_to_fear | text | Exact span (if G=1) |
| N: span_causal_oversimplification | text | Exact span (if H=1) |
| O: span_flag_waving | text | Exact span (if I=1) |
| P: notes | text | Optional: doubts, edge cases, reasoning |

#### Tab 3: "annotator_gustavo" (Gustavo fills this)

Same structure as Tab 2.

### Why this structure?

- **Binary columns (0/1) per technique** — makes computing Cohen's κ trivial (just compare columns D-I between annotators)
- **Separate span columns** — the span is evidence for why you labeled 1, and will be used to compare against model predictions
- **Notes column** — captures uncertainty for discussion during resolution
- **has_technique column** — quick sanity check (should be YES if any of D-I is 1)

---

## 3. Annotation Workflow

### Phase 1: Calibration (repeat until κ ≥ 0.40 per technique)

1. **Round 1**: Both annotators independently label the same 50 segments
2. **Compute agreement**: Run the agreement script to get κ per technique
3. **Discussion session**: Meet to discuss disagreements. Focus on:
   - Cases where one said YES and other said NO
   - Understanding WHY you disagreed (definition ambiguity? missed span? different interpretation?)
4. **Refine guidelines**: If a systematic disagreement pattern emerges, add clarification to Section 4
5. **Round 2**: Label a NEW set of 50 segments (never re-use calibration segments)
6. **Repeat** until all techniques have κ ≥ 0.40

**Important**: Calibration segments are EXCLUDED from the final evaluation set.

### Phase 2: Main Annotation

1. Select the annotation set (e.g., 400-600 segments)
2. Split into batches of ~100 segments each
3. Both annotators independently label all segments (same set, no splitting)
4. ~20% of segments must overlap (for computing final agreement)
5. The rest can be split between annotators for efficiency

### Phase 3: Disagreement Resolution

1. Identify all segments where annotators disagree
2. Meet to discuss each disagreement
3. Reach consensus → this becomes the **gold label**
4. Record the resolution in a "consolidated" file

---

## 4. Technique Definitions and Annotation Guidelines

### 4.1 Loaded Language

**Definition**: Use of words or expressions with disproportionate emotional charge (positive or negative) to influence the audience through emotion rather than reason.

**Label as Loaded Language when**:
- A word/expression has stronger emotional weight than the context requires
- The emotional language is used to persuade, not just to describe
- Replacing the word with a neutral synonym would change the persuasive effect

**Do NOT label when**:
- The emotional language is proportionate to the situation (e.g., "catástrofe" when describing actual flooding with hundreds dead)
- The speaker is quoting someone else
- Standard parliamentary formality (e.g., "nobre deputado", "esta Casa")

**Examples (LABEL)**:
- "Essa política **assassina** de empregos" — "assassina" is disproportionately violent
- "O governo **irresponsável** está **destruindo** o futuro" — emotional amplification
- "A **máquina de corrupção** instalada em Brasília" — emotive metaphor

**Examples (DO NOT LABEL)**:
- "O governo implementou uma nova política econômica" — neutral
- "A inflação está alta" — factual, even if negative
- "Muito obrigado, Sr. Presidente" — formality
- "Infelizmente, não conseguimos aprovar" — mild, proportionate

### 4.2 Name Calling

**Definition**: Assigning pejorative labels, insults, or stereotypes to a political opponent to discredit them without addressing the merit of their argument.

**Label when**:
- A person or group is labeled with a derogatory term
- The label is used to dismiss rather than engage with the argument
- The focus is on attacking the person, not their position

**Do NOT label when**:
- Using the actual name of a party or ideology without pejorative intent (e.g., "a esquerda" or "a direita" as neutral descriptors)
- Criticizing actions rather than assigning labels (e.g., "o governo errou" is not name calling)

**Examples (LABEL)**:
- "Esses **comunistas** querem destruir a família" — pejorative label
- "Os **fascistas** no poder" — labeling to discredit
- "Esse **bando de ladrões**" — collective insult

**Examples (DO NOT LABEL)**:
- "O deputado da oposição apresentou uma proposta diferente" — neutral reference
- "Discordamos da posição do governo" — criticizing position, not naming

### 4.3 Doubt

**Definition**: Sowing doubt about facts, institutions, or people without presenting concrete evidence, using insinuations, rhetorical questions, or vague references.

**Label when**:
- The speaker implies wrongdoing without evidence ("dizem que...", "há quem diga...")
- Rhetorical questions are used to cast suspicion ("será que podemos confiar...?")
- Vague references to unnamed sources create doubt ("pessoas ligadas a...")

**Do NOT label when**:
- Legitimate questioning with specific evidence cited
- Parliamentary oversight questioning (asking ministers for data is not "doubt")
- Questions that expect actual answers (not rhetorical)

**Examples (LABEL)**:
- "**Dizem** que esse deputado tem **ligações** com organizações criminosas" — insinuation without evidence
- "**Será que** podemos confiar nos números apresentados?" — rhetorical question casting doubt
- "**Há quem diga** que por trás dessa votação existem interesses **obscuros**" — vague insinuation

**Examples (DO NOT LABEL)**:
- "Os dados do IBGE mostram uma queda de 2%" — citing evidence
- "Gostaria de perguntar ao Ministro quantos empregos foram criados" — legitimate question

### 4.4 Appeal to Fear

**Definition**: Creating or amplifying fear through catastrophic scenarios, threats, or extreme consequences to manipulate decisions.

**Label when**:
- A catastrophic outcome is presented as inevitable if action X is (or isn't) taken
- The fear is disproportionate to the actual risk
- The argument relies on fear rather than evidence or logic

**Do NOT label when**:
- Citing real risks with proportionate language (e.g., "economists warn inflation may rise")
- Describing actual ongoing crises (e.g., floods, pandemic)
- Conditional statements with reasonable probability assessments

**Examples (LABEL)**:
- "Se aprovarmos essa lei, **o Brasil vai virar uma Venezuela**" — catastrophic scenario
- "**Vão destruir** sua família, **tirar** suas armas e seus filhos" — threat amplification
- "**Milhões perderão o emprego e passarão fome**" — extreme consequence without evidence

**Examples (DO NOT LABEL)**:
- "Economistas alertam que a inflação pode subir" — proportionate warning
- "O relatório indica riscos para o equilíbrio fiscal" — evidence-based risk

### 4.5 Causal Oversimplification

**Definition**: Attributing a single simple cause to a complex problem with multiple causes, deliberately ignoring complexity to strengthen a political argument.

**Label when**:
- A complex issue is reduced to one cause
- The oversimplification serves an argumentative purpose
- The speaker ignores obvious contributing factors

**Do NOT label when**:
- Highlighting one important cause among many (especially if acknowledged: "entre outros fatores...")
- Time-limited statements about specific proximate causes
- Simplification for brevity in a longer speech that addresses complexity elsewhere

**Examples (LABEL)**:
- "**A culpa** de toda a violência no Brasil **é** do desarmamento" — single cause
- "O desemprego existe **porque** o governo não corta gastos" — ignores multiple factors
- "**Se não fossem** os políticos corruptos, o Brasil seria primeiro mundo" — single cause fantasy

**Examples (DO NOT LABEL)**:
- "O aumento da violência tem múltiplas causas, incluindo desigualdade e desemprego" — acknowledges complexity
- "Um dos fatores que contribuiu para a crise foi..." — one among many

### 4.6 Flag-Waving

**Definition**: Appealing to patriotism, religion, group identity, or national values to justify a political position, invoking loyalty rather than rational arguments.

**Label when**:
- A political position is justified by invoking group loyalty ("quem ama o Brasil...")
- National/religious identity is used as an argument itself
- The appeal substitutes for substantive reasoning

**Do NOT label when**:
- Mentioning Brazil, God, or national values in a descriptive context
- Patriotic language that accompanies (rather than replaces) substantive arguments
- Standard parliamentary closing phrases ("Que Deus abençoe" without argumentative function)

**Examples (LABEL)**:
- "**Quem ama o Brasil de verdade** não pode aceitar essa proposta" — loyalty as argument
- "**Deus, pátria e família** — é isso que defendemos" — identity as justification
- "**Em nome do povo brasileiro** e da nossa soberania, votaremos contra" — nationalism substituting argument

**Examples (DO NOT LABEL)**:
- "Este projeto beneficia cidadãos de todas as regiões" — mentions country descriptively
- "A Constituição Federal garante esse direito" — legal reference, not identity appeal
- "Que Deus abençoe V.Exas." — formulaic closing

---

## 5. Edge Cases and Common Doubts

### "Is emotion always manipulation?"
NO. A deputy who is genuinely passionate about healthcare is not necessarily using Loaded Language. The key is whether the emotional language is **disproportionate to the context** and **used to persuade instead of inform**. "A enfermagem não aguenta mais!" from a deputy defending nurses — this is passionate but proportionate. "Política assassina de empregos" — this is disproportionate (policies don't literally kill).

### "Can a segment have both Loaded Language AND Name Calling?"
YES. "Esses comunistas irresponsáveis estão destruindo o país" has Name Calling ("comunistas" as pejorative) AND Loaded Language ("irresponsáveis", "destruindo").

### "What if I'm unsure?"
Mark what you're confident about. Use the notes column to write "UNSURE: possible Doubt?" and discuss during resolution. Better to be conservative (don't label) than to inflate counts.

### "What about quoting others?"
If the deputy is quoting someone else's manipulative language to criticize it, do NOT label. If they're approvingly repeating or endorsing the manipulation, label it.

---

## 6. Practical Tips

1. **Read the full segment first** before deciding anything. Context matters.
2. **Be conservative** — when in doubt, don't label. This reduces false positives.
3. **Work in batches** of 25-50 segments to maintain focus. Take breaks.
4. **Don't look at the party/spectrum** while annotating. Focus on the text.
5. **Use the notes column** generously — it helps during disagreement resolution.
6. **Don't discuss with your partner** until after both finish a calibration batch.

---

## 7. Export Format

After annotation is complete, export each annotator's tab as CSV, then run the conversion script:

1. Calibration round 1 (50 segments — you already did this)
uv run python scripts/export_for_annotation.py --n 50 --output annotation_calibration_round1.csv

2. Calibration round 2 (if needed — different 50 segments, different seed)
uv run python scripts/export_for_annotation.py --n 50 --seed 123 --output annotation_calibration_round2.csv

3. Main annotation batch (400 or 600 segments)
uv run python scripts/export_for_annotation.py --n 600 --seed 99 --output annotation_main.csv


```bash
uv run python scripts/convert_annotations.py --annotator brenno --input annotations_brenno.csv
uv run python scripts/convert_annotations.py --annotator gustavo --input annotations_gustavo.csv
```

This produces JSONL files compatible with the evaluation pipeline:
- `data/annotations/main/annotator_a.jsonl`
- `data/annotations/main/annotator_b.jsonl`

---

## 8. Timeline

| Phase | Segments | Estimated Time | Output |
|-------|----------|---------------|--------|
| Calibration Round 1 | 50 | ~2h each | κ per technique |
| Discussion + Guideline refinement | — | 1h meeting | Updated guidelines |
| Calibration Round 2 (if needed) | 50 | ~2h each | κ per technique |
| Main annotation | 400-600 | ~12-15h each | Full labels |
| Disagreement resolution | varies | ~3-5h meeting | Gold labels |

Total per annotator: ~15-20 hours spread over 2-3 weeks.
