"""Local, source-grounded summarization helpers.

Every summary, point, question, and comparison is derived from the source string
passed to create_report. No UI or prior report content is read here.
"""
import os
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path

SAMPLE_ARTICLE = """Artificial intelligence (AI) is increasingly used in education to support learning and teaching. Personalized learning systems analyze a learner's progress and adapt the pace, practice, or materials to their needs. This can help students receive timely support, while teachers gain information about areas where a class may need more instruction.

Automated assessment tools can score selected assignments and provide quick feedback. Intelligent tutoring systems offer guided practice, hints, and explanations as students work through a problem. These tools may make practice more available, but their feedback should be checked for accuracy and should not replace human judgment in important decisions.

AI can improve access to learning resources, help educators notice patterns in student performance, and reduce some repetitive administrative work. However, these benefits depend on careful design and meaningful oversight. Systems trained on incomplete or unrepresentative data may produce biased recommendations or treat groups unfairly. Schools should examine performance across student groups and provide ways to question or correct automated decisions.

Privacy and data security are central concerns because educational systems may process sensitive information about children, learning needs, and performance. Institutions should collect only necessary data, explain how it is used, protect it from unauthorized access, and follow applicable privacy requirements. Students and families should understand the role of AI in decisions that affect them.

Teachers remain essential. They interpret students' needs, build relationships, create supportive classrooms, and bring context and professional judgment that automated systems lack. AI is most useful as a tool that supports teachers and learners rather than as a replacement for educators.

Challenges include unequal access to devices and connectivity, staff training, the cost of implementation, integration with existing curricula, and the risk of overreliance on automated recommendations. Responsible implementation therefore requires clear educational goals, human oversight, transparent communication, privacy safeguards, bias monitoring, and regular evaluation of whether the tools improve learning for all students."""

STOP_WORDS = set("a an the and or but if while of to in on for from with by as is are was were be been being this that these those it its their they them we our you your can may should would could will do does did have has had not than then into about across through more most some such each both also very increasingly selected important central however therefore because rather where when what how who which whose there here document documents study studies paper research use uses used using based may might many much one two three include includes included only collect collects collected create creates created notice notices receive receives require requires required produce produces treat treats treated process processes processed offer offers offered follow follows followed remain remains remained applicable provide provides provided replace replaces replaced gain gains gained depend depends depended bring brings brought also".split())
WORD_RE = re.compile(r"[A-Za-z][A-Za-z'-]*")
SENTENCE_RE = re.compile(r"(?<=[.!?])(?:[\"'”’)]*)\s+(?=[\"'“‘(]*[A-Z0-9])")


def normalize_text(text):
    """Normalize whitespace without joining sentence boundaries."""
    text = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"[\t\f\v ]+", " ", line).strip() for line in text.split("\n")]
    paragraphs = []
    for line in lines:
        if line:
            paragraphs.append(line)
        elif paragraphs and paragraphs[-1] != "":
            paragraphs.append("")
    return "\n".join(paragraphs).strip()


def sentences(text):
    normalized = normalize_text(text).replace("\n", " ")
    normalized = re.sub(r"\s+", " ", normalized).strip()
    if not normalized:
        return []
    # Retain short but meaningful sentences; headings are filtered separately.
    return [part.strip() for part in SENTENCE_RE.split(normalized) if WORD_RE.search(part)]


def tokens(text):
    return [w.lower().strip("'-") for w in WORD_RE.findall(text)
            if len(w.strip("'-")) > 2 and w.lower() not in STOP_WORDS]


def _content_sentences(text):
    """Remove title-like fragments and repeated sentences while preserving order."""
    result, seen = [], set()
    for sentence in sentences(text):
        words = sentence.split()
        if len(words) < 5:
            continue
        key = re.sub(r"\W+", " ", sentence.lower()).strip()
        if key and key not in seen:
            seen.add(key)
            result.append(sentence)
    return result


def _sentence_scores(parts):
    docs = [tokens(s) for s in parts]
    df = Counter(word for terms in docs for word in set(terms))
    n = max(1, len(parts))
    scores = []
    for index, terms in enumerate(docs):
        if not terms:
            scores.append(0.0)
            continue
        tf = Counter(terms)
        score = sum((1 + (count - 1) * .25) * (1 + (n / (1 + df[word]))) for word, count in tf.items())
        score /= len(terms) ** .65
        # Mild lead bias, not enough to override topic relevance.
        score *= 1.12 if index == 0 else (1.05 if index < 3 else 1.0)
        scores.append(score)
    return scores


def select_sentences(text, target_words):
    parts = _content_sentences(text)
    if not parts:
        return []
    scores = _sentence_scores(parts)
    max_words = 100 if target_words <= 100 else 300
    first_terms = set(tokens(parts[0]))
    selected, selected_terms, count = [(0, parts[0], first_terms)], [first_terms], len(parts[0].split())
    for index in sorted(range(1, len(parts)), key=lambda i: (-scores[i], i)):
        sentence = parts[index]
        terms = set(tokens(sentence))
        # Penalize redundant sentences that repeat most of the ideas already selected.
        redundancy = max((len(terms & old) / max(1, len(terms | old)) for old in selected_terms), default=0)
        adjusted = scores[index] * (1 - .72 * redundancy)
        if count >= target_words or adjusted < scores[index] * .30:
            continue
        if count + len(sentence.split()) > max_words:
            continue
        selected.append((index, sentence, terms))
        selected_terms.append(terms)
        count += len(sentence.split())
        if count >= target_words:
            break
    return [sentence for _, sentence, _ in sorted(selected)]


def extractive_summary(text, target_words):
    return " ".join(select_sentences(text, target_words))


@lru_cache(maxsize=1)
def _local_transformer():
    """Load a user-provided local model only; never access the network."""
    model_path = os.environ.get("SUMMARIZER_MODEL_PATH", "").strip()
    if not model_path or not Path(model_path).is_dir():
        return None
    try:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path, local_files_only=True)
        return pipeline("summarization", model=model, tokenizer=tokenizer)
    except Exception:
        return None


def make_summary(text, target_words):
    model = _local_transformer()
    if model:
        try:
            result = model(text[:12000], max_length=max(40, target_words * 2),
                           min_length=min(35, max(10, target_words // 2)), truncation=True)
            generated = re.sub(r"\s+", " ", result[0].get("summary_text", "")).strip()
            if generated:
                return generated, "Local Hugging Face Transformers model"
        except Exception:
            pass
    return extractive_summary(text, target_words), "Extractive frequency-based summarization (local fallback)"


def _phrase_candidates(text):
    phrases = Counter()
    for sentence in sentences(text):
        current, previous_end = [], 0
        for match in WORD_RE.finditer(sentence):
            word = match.group(0)
            normalized = word.lower().strip("'-")
            separator = sentence[previous_end:match.start()]
            if re.search(r"[,;:.!?()\u2013\u2014]", separator) and current:
                _add_phrase_chunks(phrases, current)
                current = []
            previous_end = match.end()
            if len(normalized) <= 2 or normalized in STOP_WORDS:
                if current:
                    _add_phrase_chunks(phrases, current)
                    current = []
            else:
                current.append(normalized)
        if current:
            _add_phrase_chunks(phrases, current)
    return phrases


def _add_phrase_chunks(phrases, current):
    for size in (3, 2, 1):
        for start in range(len(current) - size + 1):
            phrase = tuple(current[start:start + size])
            if not (len(phrase) == 1 and len(phrase[0]) < 4):
                phrases[phrase] += 1


def keyword_list(text, limit=10):
    candidates = _phrase_candidates(text)
    if not candidates:
        return []
    word_frequency = Counter(tokens(text))
    def keyword_score(item):
        phrase, count = item
        specificity = sum(1 / max(1, word_frequency[word]) ** .5 for word in phrase) / len(phrase)
        phrase_bonus = 2.0 if len(phrase) > 1 else 1.0
        return count * (len(phrase) ** .9) * specificity * phrase_bonus
    ranked = sorted(candidates.items(), key=lambda item: (
        -keyword_score(item),
        -len(item[0]), " ".join(item[0])))
    chosen = []
    for phrase, _ in ranked:
        value = " ".join(phrase)
        # Avoid listing both a phrase and its contained unigrams as separate tags.
        if any(value in existing or existing in value for existing in chosen):
            continue
        chosen.append(value)
        if len(chosen) >= limit:
            break
    # Fill remaining slots with distinct content words when the text has enough.
    if len(chosen) < min(5, len(candidates)):
        for phrase, _ in ranked:
            value = " ".join(phrase)
            if value not in chosen:
                chosen.append(value)
            if len(chosen) >= min(limit, max(5, len(candidates))):
                break
    return chosen[:limit]


def main_points(text, limit=8):
    parts = _content_sentences(text)
    if not parts:
        return []
    scores = _sentence_scores(parts)
    ranked = sorted(range(len(parts)), key=lambda i: (-scores[i], i))
    chosen, chosen_sets = [], []
    for index in ranked:
        terms = set(tokens(parts[index]))
        if len(terms) < 3:
            continue
        similarity = max((len(terms & other) / max(1, len(terms | other)) for other in chosen_sets), default=0)
        if similarity >= .68:
            continue
        chosen.append(index)
        chosen_sets.append(terms)
        if len(chosen) >= min(limit, 10):
            break
    return [parts[i] for i in sorted(chosen)]


def conclusions(text, points, limit=4):
    parts = _content_sentences(text)
    markers = re.compile(r"\b(therefore|conclude|conclusion|requires?|depends|should|must|essential|important|result|finding|demonstrat|suggest|indicat|benefit|risk|challenge|recommend)\w*\b", re.I)
    candidates = [s for s in parts if markers.search(s)]
    output = []
    for sentence in candidates:
        if sentence not in output:
            output.append(sentence)
        if len(output) >= limit:
            break
    # Do not invent conclusions to reach a minimum count.
    return output


def revision_questions(text, points, keywords):
    source_points = points or _content_sentences(text) or sentences(text) or [normalize_text(text)]
    source_points = [point for point in source_points if point] or [normalize_text(text)]
    topic = keywords[0] if keywords else next(iter(tokens(text)), "the topic")
    candidates = [
        f"Definition-based: What does the source explain about {topic}?",
        f"Conceptual: How does {topic} relate to the document's main subject?",
        f"Comprehension: What is the main claim in this source point: {source_points[0]}",
        f"Application: Which action described in the document could be applied to {topic}?",
        f"Analytical: What evidence or limitation in the document qualifies its discussion of {topic}?",
    ]
    # For shorter source material, anchor each question to a different available
    # sentence where possible, while retaining the requested five question types.
    result, seen = [], set()
    for index, question in enumerate(candidates):
        anchor = source_points[min(index, len(source_points) - 1)]
        if index == 2:
            question = f"Comprehension: What is the main claim in this source point: {anchor}"
        elif index == 4:
            question = f"Analytical: What evidence or limitation does this point describe: {anchor}"
        normalized = re.sub(r"\W+", " ", question.lower()).strip()
        if normalized in seen:
            question = f"{question} (consider its practical implications)"
            normalized = re.sub(r"\W+", " ", question.lower()).strip()
        seen.add(normalized)
        result.append(question)
    return result


def _overlap(a, b):
    left, right = set(tokens(a)), set(tokens(b))
    return len(left & right) / max(1, len(left | right))


def compare_summary(text, summary, points):
    """Approximate lexical support and point coverage; not factual verification."""
    summary_sentences = sentences(summary)
    rows, represented = [], 0
    for point in points:
        best = max(summary_sentences, key=lambda candidate: _overlap(point, candidate), default="")
        similarity = _overlap(point, best) if best else 0.0
        point_terms = set(tokens(point))
        best_terms = set(tokens(best))
        recall = len(point_terms & best_terms) / max(1, len(point_terms))
        if not best or similarity < .08:
            accuracy = "Not Found"
        elif recall >= .68 and similarity >= .45:
            accuracy = "Supported"
        elif recall >= .25:
            accuracy = "Partially Supported"
        else:
            accuracy = "Not Verifiable"
        coverage = "Represented" if recall >= .55 else "Partially Represented" if recall >= .25 else "Not Represented"
        represented += recall >= .55
        rows.append({"point": point, "statement": best or "No matching statement found.",
                     "accuracy": accuracy, "coverage": coverage, "overlap": round(similarity, 3)})
    percentage = round(100 * represented / len(points)) if points else 0
    return rows, percentage


def build_report_text(report):
    lines = [f"AI-Powered Text Summarization Report\nDocument title: {report['title']}",
             "\nShort summary", report["short_summary"], "\nDetailed summary"]
    for heading, body in report["detailed_summary"].items():
        if body:
            lines.extend([heading, body])
    lines.extend(["\nMain points"])
    lines.extend(f"{i}. {point}" for i, point in enumerate(report["main_points"], 1))
    lines.extend(["\nKeywords", ", ".join(report["keywords"]), "\nKey conclusions"])
    lines.extend(f"- {item}" for item in report["conclusions"])
    lines.append("\nRevision questions")
    lines.extend(f"{i}. {question}" for i, question in enumerate(report["questions"], 1))
    lines.extend(["\nAccuracy and coverage (approximate lexical checks; not factual verification)"])
    for kind in ("short", "detailed"):
        represented = report["coverage_basis"][f"{kind}_represented"]
        total = report["coverage_basis"]["source_points"]
        lines.extend([f"{kind.title()} summary coverage: {report['coverage'][kind]}% ({represented}/{total} source points)",
                      f"{kind.title()} summary comparison:"])
        lines.extend(f"- Original point: {row['point']} | Summary statement: {row['statement']} | Accuracy: {row['accuracy']} | Coverage: {row['coverage']}"
                     for row in report["accuracy"][kind])
    lines.extend(["\nSummarization method", report["method"]])
    return "\n".join(lines).strip() + "\n"


def create_report(text, title="Untitled document", length="short"):
    source = normalize_text(text)
    points = main_points(source)
    keywords = keyword_list(source, 10)
    short, short_method = make_summary(source, 80)
    detailed, detailed_method = make_summary(source, 230)
    detailed_sentences = sentences(detailed)
    # Divide the unique selected detailed-summary sentences into distinct
    # sections. No sentence is copied into multiple headings.
    if not detailed_sentences:
        detailed_sentences = select_sentences(source, 230)
    chunks = [detailed_sentences[:1], detailed_sentences[1:-2], detailed_sentences[-2:-1], detailed_sentences[-1:]]
    section_names = ("Introduction", "Main Discussion", "Important Findings", "Conclusion")
    sections = {name: " ".join(chunk) for name, chunk in zip(section_names, chunks)}
    questions = revision_questions(source, points, keywords)
    key_conclusions = conclusions(source, points)
    short_rows, short_coverage = compare_summary(source, short, points)
    detailed_rows, detailed_coverage = compare_summary(source, detailed, points)
    coverage_basis = {
        "source_points": len(points),
        "short_represented": sum(row["coverage"] == "Represented" for row in short_rows),
        "detailed_represented": sum(row["coverage"] == "Represented" for row in detailed_rows),
    }
    detailed_display = "\n\n".join(f"{heading}: {body}" for heading, body in sections.items() if body)
    report = {"title": title, "length": length,
              "summary": short if length == "short" else detailed_display,
              "short_summary": short, "detailed_text": detailed,
              "detailed_summary": sections, "main_points": points,
              "keywords": keywords, "conclusions": key_conclusions,
              "questions": questions,
              "accuracy": {"short": short_rows, "detailed": detailed_rows},
              "coverage": {"short": short_coverage, "detailed": detailed_coverage},
              "coverage_basis": coverage_basis,
              "method": f"Short: {short_method} | Detailed: {detailed_method}"}
    report["report_text"] = build_report_text(report)
    return report
