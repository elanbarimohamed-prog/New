import math
from datetime import datetime, timezone
from app.core.models import (
    ScoredArticle,
    ObservedFact,
    PipelineRationale,
    DecisionVerdict,
    PipelineMode,
)
from app.core.session_window import SessionWindow
from app.core.exceptions import FreshnessViolationException
from app.core.constants import DIRECTIONAL_THRESHOLD, DEFAULT_WEIGHT_FLOOR

# 14-hour half life for exponential recency decay
LAMBDA_DECAY = math.log(2.0) / 14.0


class AggregatorService:
    @classmethod
    def aggregate_and_decide(
        cls,
        ticker: str,
        company_name: str,
        session: SessionWindow,
        articles: list[ScoredArticle],
        observed_facts: list[ObservedFact],
        llm_summary: str = "",
        pipeline_mode: PipelineMode = PipelineMode.HYBRID_FINBERT_LLM,
    ) -> tuple[DecisionVerdict, float, float, float, PipelineRationale, list[str]]:
        """
        Calculates:
        - Exponential recency decay
        - Composite sentiment score S in [-1.0, 1.0] using effective weights
        - Evidence quality Q in [0.0, 1.0]
        - Decision confidence C in [0.0, 1.0]
        - Enforces the 2-distinct-publisher corroboration rule for BUY/SELL
        - Heuristic decision matrix verdict
        - Plain-English rationale separating facts vs inferences
        
        Returns:
        (decision, confidence, sentiment_score, evidence_quality, rationale, warnings)
        """
        warnings: list[str] = []
        if pipeline_mode == PipelineMode.DETERMINISTIC_FINBERT_ONLY:
            warnings.append("LLM synthesis unavailable: Result generated using local NLP classifier only (confidence capped at 0.60).")

        # 1. Filter valid non-excluded articles
        valid_articles = [a for a in articles if a.exclusion_reason is None]

        # 2. Defense-in-depth: enforce core invariant that all valid articles are strictly inside session window
        in_window_articles: list[ScoredArticle] = []
        for art in valid_articles:
            if art.published_at_utc is None or not session.is_timestamp_in_window(art.published_at_utc):
                raise FreshnessViolationException(message=
                    f"Core Invariant Violated in Aggregator: Article '{art.article_id}' has invalid timestamp {art.published_at_utc} outside session window [{session.window_start_utc}, {session.window_end_utc}]."
                )
            in_window_articles.append(art)

        # Check for zero news
        if len(in_window_articles) == 0:
            summary = f"No verifiable news found for {ticker} within the current trading session window."
            rationale = PipelineRationale(
                summary=summary,
                observed_facts=[],
                inferences=["Zero articles passed session window and entity relevance filters."],
            )
            return (
                DecisionVerdict.HOLD,
                0.0,
                0.0,
                0.0,
                rationale,
                warnings + ["No current-day news available within the session window."]
            )

        # 3. Compute recency weights and directional tallies consistently (F-08)
        now_utc = session.run_timestamp_utc
        pos_count = 0
        neg_count = 0
        neu_count = 0

        valid_publishers = set()
        valid_clusters = set()

        directional_weighted_sum = 0.0
        directional_weight_sum = 0.0

        for art in in_window_articles:
            pub_utc = art.published_at_utc
            if pub_utc.tzinfo is None:
                pub_utc = pub_utc.replace(tzinfo=timezone.utc)
            delta_hours = max(0.0, (now_utc - pub_utc).total_seconds() / 3600.0)
            recency_decay = math.exp(-LAMBDA_DECAY * delta_hours)

            # Compute effective_weight once per article
            effective_weight = round(recency_decay * max(DEFAULT_WEIGHT_FLOOR, art.final_weight), 4)
            art.effective_weight = effective_weight
            art.age_hours = round(delta_hours, 2)

            # Skip articles whose effective weight is not positive
            if effective_weight <= 0.0:
                continue

            # Directional tally and scoring
            if art.finbert_directional_score >= DIRECTIONAL_THRESHOLD:
                pos_count += 1
                directional_weighted_sum += effective_weight * art.finbert_directional_score
                directional_weight_sum += effective_weight
                if art.canonical_publisher and art.canonical_publisher != "Unknown Aggregator":
                    valid_publishers.add(art.canonical_publisher)
                if art.cluster_id:
                    valid_clusters.add(art.cluster_id)
            elif art.finbert_directional_score <= -DIRECTIONAL_THRESHOLD:
                neg_count += 1
                directional_weighted_sum += effective_weight * art.finbert_directional_score
                directional_weight_sum += effective_weight
                if art.canonical_publisher and art.canonical_publisher != "Unknown Aggregator":
                    valid_publishers.add(art.canonical_publisher)
                if art.cluster_id:
                    valid_clusters.add(art.cluster_id)
            else:
                neu_count += 1

        # 4. Composite Sentiment Score S in [-1.0, 1.0] (computed from directional evidence)
        if directional_weight_sum > 0:
            sentiment_score = round(max(-1.0, min(1.0, directional_weighted_sum / directional_weight_sum)), 4)
        else:
            sentiment_score = 0.0

        # 5. Evidence Quality Metric Q in [0.0, 1.0]
        distinct_pub_count = len(valid_publishers)
        distinct_cluster_count = len(valid_clusters)
        n_valid = pos_count + neg_count + neu_count

        directional_total = pos_count + neg_count

        volume_factor = min(1.0, max(0.3, directional_total / 3.0))
        diversity_factor = 0.5 + 0.5 * min(1.0, max(1, distinct_pub_count) / 2.0)
        evidence_quality = round(min(1.0, max(0.1, volume_factor * diversity_factor)), 4)

        # 6. Decision Confidence C in [0.0, 1.0]
        if directional_total > 0:
            consensus_ratio = abs(pos_count - neg_count) / directional_total
        else:
            consensus_ratio = 0.0

        raw_confidence = abs(sentiment_score) * (0.4 + 0.6 * consensus_ratio) * evidence_quality
        if pipeline_mode == PipelineMode.DETERMINISTIC_FINBERT_ONLY:
            decision_confidence = round(min(0.60, max(0.15, raw_confidence)), 4)
        else:
            decision_confidence = round(min(0.95, max(0.20, raw_confidence)), 4)

        # 7. Evaluate Ratio-Based Decision Matrix & Publisher Corroboration (F-09)
        inferences = [
            f"FinBERT evaluated {n_valid} total headlines: {pos_count} positive, {neg_count} negative, {neu_count} neutral.",
            f"Calculated directional sentiment score: {sentiment_score:+.2f} (Neutral articles excluded from score dilution).",
            f"Evidence quality: {evidence_quality:.2f} across {distinct_pub_count} distinct publisher(s) and {distinct_cluster_count} cluster(s).",
            f"Decision confidence: {decision_confidence:.2f} (Directional consensus ratio: {consensus_ratio:.2f}).",
        ]

        # Rule A: Sparse Evidence
        if directional_total < 2 or n_valid < 2:
            decision = DecisionVerdict.HOLD
            summary = f"HOLD: Insufficient directional headline volume for {ticker} ({directional_total} directional headline(s)). Sparse evidence mandates a neutral stance."
            inferences.append("Decision rule applied: Minimum directional evidence threshold not met (mandates HOLD).")

        # Rule B: Strong Positive BUY (Ratio Pos >= 1.5 * Neg and Pos >= 2)
        elif pos_count >= 2 and (neg_count == 0 or pos_count >= 1.5 * neg_count) and sentiment_score >= 0.20:
            if distinct_pub_count < 2:
                decision = DecisionVerdict.HOLD
                summary = f"HOLD: Positive directional sentiment observed ({pos_count} positive) but lacks multi-publisher corroboration ({distinct_pub_count} publisher(s))."
                warnings.append("INSUFFICIENT_CORROBORATION")
                inferences.append("Corroboration rule applied: BUY requires at least 2 distinct canonical publishers; single-source evidence forces HOLD.")
            else:
                decision = DecisionVerdict.BUY
                summary = f"BUY: Positive catalyst momentum supported by {pos_count} positive headlines vs {neg_count} negative across {distinct_pub_count} publishers (Pos/Neg ratio: {pos_count}/{max(1, neg_count)})."
                inferences.append(f"Decision rule applied: Positive ratio threshold met ({pos_count} pos >= 1.5 * {neg_count} neg, S={sentiment_score:+.2f} >= +0.20, corroborated by {distinct_pub_count} publishers).")

        # Rule C: Strong Negative SELL (Ratio Neg >= 1.5 * Pos and Neg >= 2)
        elif neg_count >= 2 and (pos_count == 0 or neg_count >= 1.5 * pos_count) and sentiment_score <= -0.20:
            if distinct_pub_count < 2:
                decision = DecisionVerdict.HOLD
                summary = f"HOLD: Negative directional sentiment observed ({neg_count} negative) but lacks multi-publisher corroboration ({distinct_pub_count} publisher(s))."
                warnings.append("INSUFFICIENT_CORROBORATION")
                inferences.append("Corroboration rule applied: SELL requires at least 2 distinct canonical publishers; single-source evidence forces HOLD.")
            else:
                decision = DecisionVerdict.SELL
                summary = f"SELL: Negative catalyst pressure supported by {neg_count} negative headlines vs {pos_count} positive across {distinct_pub_count} publishers (Neg/Pos ratio: {neg_count}/{max(1, pos_count)})."
                inferences.append(f"Decision rule applied: Negative ratio threshold met ({neg_count} neg >= 1.5 * {pos_count} pos, S={sentiment_score:+.2f} <= -0.20, corroborated by {distinct_pub_count} publishers).")

        # Rule D: Conflicting Evidence
        elif directional_total >= 2 and pos_count >= 1 and neg_count >= 1:
            decision = DecisionVerdict.HOLD
            summary = f"HOLD: Conflicting market sentiment ({pos_count} positive vs. {neg_count} negative catalysts). Balanced opposing forces indicate neutral market context."
            inferences.append("Decision rule applied: Positive and negative catalysts balance within 1.5x ratio corridor.")

        # Rule E: Default Neutral HOLD
        else:
            decision = DecisionVerdict.HOLD
            summary = f"HOLD: Market sentiment for {ticker} is balanced or within neutral thresholds (S={sentiment_score:+.2f})."
            inferences.append(f"Decision rule applied: Sentiment within neutral corridor (S={sentiment_score:+.2f}).")

        rationale = PipelineRationale(
            summary=summary,
            observed_facts=observed_facts,
            inferences=inferences,
        )

        return (
            decision,
            decision_confidence,
            sentiment_score,
            evidence_quality,
            rationale,
            warnings,
        )


