import json
import logging
from pathlib import Path
from typing import Literal

import faiss
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel

from config import llm


logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s : %(message)s"
)


metrics_file = "data/metrics.json"
index_directory = "data/transcript_index"
results_directory = "results"


class EvaluationResult(BaseModel):
    result: Literal["PASS", "FAIL", "NOT_APPLICABLE"]
    justification: str
    evidence: str


class EmpathyEvidence(BaseModel):
    customer: str
    agent: str


class EmpathyEvaluationResult(BaseModel):
    result: Literal["PASS", "FAIL", "NOT_APPLICABLE"]
    justification: str
    evidence: list[EmpathyEvidence]


EMPATHY_EXAMPLES = """
Examples of empathetic customer-service statements. These are examples of
language that may demonstrate empathy. Do not require exact wording; look
for semantically equivalent expressions in the actual transcript.

Acknowledging Frustration:
- I can understand how frustrating that must be.
- I'm really sorry this has been such a hassle.
- I see why that would be upsetting.
- That definitely shouldn't have happened.
- I can hear how important this is to you.
- I understand how inconvenient this has been.
- That sounds incredibly frustrating.
- I appreciate you explaining what happened.
- I can tell this has been a tough experience.
- You shouldn't have had to deal with that.

Validating Feelings:
- It makes sense that you feel that way.
- Anyone in your position would feel similarly.
- I would feel the same way.
- Your frustration is completely understandable.
- You have every right to be concerned.
- I can see why this raised questions.
- That's a reasonable concern.
- I understand why that would be disappointing.
- That reaction makes sense.
- Thanks for being honest about how you're feeling.

When a Customer Is Angry:
- I can hear how upsetting this has been.
- I'm sorry we've caused you stress.
- Let's work through this together.
- You shouldn't have had to experience that.
- I want to make this right.
- I understand this isn't what you expected.
- Thank you for sticking with us.
- I appreciate you bringing this to our attention.
- Let's slow this down and find a solution.
- I can see how that would be infuriating.

When There's a Delay:
- I know waiting isn't ideal.
- Thank you for your patience.
- I understand timing is important.
- I know this delay is inconvenient.
- We value your time.
- I'm sorry this is taking longer than expected.
- Let me update you on where things stand.
- I appreciate your understanding.
- Thanks for bearing with us.
- I know you were expecting this sooner.

When You Made a Mistake:
- That's on us.
- We should have handled that better.
- I'm sorry we dropped the ball.
- Thank you for pointing that out.
- You're absolutely right.
- We appreciate your honesty.
- That's valuable feedback.
- I understand why that caused confusion.
- We missed that.
- I apologize for the oversight.

When You Can't Say Yes:
- I understand why you'd ask for that.
- I wish I had more flexibility here.
- Here's what I can do.
- While I can't make that change, I want to help.
- Let's explore alternatives.
- That's a fair request.
- I see what you're hoping for.
- Let me explain why this policy exists.
- I understand this isn't the answer you wanted.
- Let's find the best available option.

When a Customer Is Confused:
- I can see how that would be confusing.
- That's not always intuitive.
- You're not alone in wondering that.
- Let me break that down clearly.
- That's a great question.
- I appreciate you double-checking.
- Let's walk through it step by step.
- Thanks for asking for clarification.
- I can see how that might be unclear.
- Let's simplify this.

When a Customer Is Urgent:
- I understand this is time-sensitive.
- Let's prioritize this.
- I can see why this feels urgent.
- Thank you for flagging this quickly.
- We'll move as fast as possible.
- I know you need a resolution quickly.
- Let me escalate this.
- I understand this impacts your work.
- We'll treat this with urgency.
- I'm on this now.

When a Customer Is Following Up:
- I'm sorry you had to follow up.
- You shouldn't have needed to reach out again.
- Thank you for your persistence.
- Let's fully resolve this now.
- I appreciate your patience.
- I can see why this feels repetitive.
- You deserve a clear answer.
- Thanks for checking back in.
- I understand the frustration of repeated contact.
- Let's close the loop properly.

When a Customer Is Disappointed:
- I'm sorry this didn't meet expectations.
- That's understandably disappointing.
- I can see why you hoped for more.
- We aim to do better than this.
- Thank you for giving us a chance to improve.
- You deserved a smoother experience.
- That's not our standard.
- I appreciate your honesty.
- I understand why this feels frustrating.
- We want to earn your trust back.

When Closing the Conversation:
- I'm glad we were able to work through this.
- Thank you for your understanding.
- I appreciate your patience today.
- Please reach out if anything else comes up.
- We're here to help.
- Thanks again for giving us the opportunity.
- I'm happy we found a solution.
- We value your business.
- Don't hesitate to contact us again.
- Your feedback truly helps.
"""


def load_metrics() -> list:
    logger.info("loading metrics....")

    with open(
        metrics_file,
        "r",
        encoding="utf-8"
    ) as file:

        data = json.load(file)

    return data["metrics"]


def load_transcript_index(transcript_path: str) -> tuple:

    recording_name = Path(transcript_path).stem

    index_directory = (
        Path("data/transcript_index") /
        recording_name
    )

    index_file = index_directory / "index.faiss"
    metadata_file = index_directory / "metadata.json"

    logger.info(
        "loading transcript index for : %s",
        recording_name
    )

    index = faiss.read_index(
        str(index_file)
    )

    with open(
        metadata_file,
        "r",
        encoding="utf-8"
    ) as file:

        metadata = json.load(file)

    return index, metadata


def load_embedding_model():

    logger.info(
        "loading embedding model..."
    )

    model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    return model


def search_chunks(
    model,
    index,
    metadata,
    criteria
) -> list:

    logger.info(
        "searching transcript for criteria: %s",
        criteria
    )

    query_embedding = model.encode(
        [criteria]
    )

    distances, indexes = index.search(
        query_embedding,
        2
    )

    retrieved_chunks = []

    for distance, chunk_index in zip(
        distances[0],
        indexes[0]
    ):

        chunk = metadata[chunk_index]

        retrieved_chunks.append(
            {
                "chunk_id": chunk["chunk_id"],
                "distance": float(distance),
                "text": chunk["text"]
            }
        )

    return retrieved_chunks


def get_all_transcript_chunks(
    metadata: list
) -> list:
    """
    Return every transcript chunk for metrics that need
    full-conversation context, such as Empathy.
    """
    logger.info(
        "Using all transcript chunks for full-conversation evaluation."
    )

    all_chunks = []

    for chunk in metadata:

        all_chunks.append(
            {
                "chunk_id": chunk["chunk_id"],
                "distance": None,
                "text": chunk["text"]
            }
        )

    return all_chunks


def is_empathy_metric(
    metric: dict
) -> bool:
    """
    Check whether the metric is the special Empathy metric.
    The comparison is case-insensitive and ignores surrounding spaces.
    """
    metric_name = str(
        metric.get("name", "")
    ).strip().lower()

    return metric_name == "empathy"


def create_prompt(
    metric: dict,
    retrieved_chunks: list
) -> str:

    criteria = metric["criteria"][0]

    prompt = f"""

You are a customer support call quality auditor.

Your task is to evaluate whether the customer support agent
satisfied the given metric.

Metric Name:
{metric["name"]}

Criteria:
{criteria}

Metric Type:
{metric["metric_type"]}

Below are the relevant parts of the call transcript.

"""

    for chunk in retrieved_chunks:

        prompt += f"""
Transcript chunk:
{chunk["text"]}
"""

    prompt += """

Evaluate the metric using only the transcript evidence provided.

Rules:

1. PASS if the transcript contains direct evidence that the agent asked
   for the information described by the criteria.

2. Treat semantically equivalent wording as satisfying the criteria.
   For example:
   - "May I have your name?" satisfies a criterion asking for the
     customer's first/last/full name.
   - "Do you have an email address?" satisfies a criterion asking
     whether the agent asked for email.

3. Do not require the agent to use the exact wording from the criteria.

4. FAIL only when the transcript shows that the agent did not ask for
   the required information.

5. NOT_APPLICABLE should be used only when the metric genuinely cannot
   be evaluated from the provided evidence.

6. Prefer direct transcript evidence over assumptions or interpretation.

7. The agent's question is sufficient evidence of satisfying an
   "asked by agent" metric; the customer's response is additional
   supporting evidence.

8. The evidence field must contain the relevant statement from the
   transcript that supports the decision.

9. The justification should be short and explain why the metric passed,
   failed, or could not be evaluated.
"""

    return prompt


def create_empathy_prompt(
    metric: dict,
    transcript_chunks: list
) -> str:
    """
    Create the special Empathy prompt using the complete transcript.
    """
    criteria = metric["criteria"][0]

    prompt = f"""

You are a customer support call quality auditor specializing in empathy.

Your task is to evaluate whether the customer support agent demonstrated
empathy when the conversation contained a genuine opportunity to do so.

Metric Name:
{metric["name"]}

Criteria:
{criteria}

Metric Type:
{metric["metric_type"]}

IMPORTANT:
This is a full-conversation empathy evaluation. The empathy opportunity
can happen anywhere in the call, so all transcript chunks are provided.
Read the complete transcript and evaluate it as one conversation.

Your task has TWO steps:

STEP 1 - FIND EMPATHY OPPORTUNITIES
Identify every meaningful situation where the customer expressed or clearly
showed frustration, anger, disappointment, confusion, urgency, concern,
inconvenience, repeated effort, or another situation where a supportive
empathetic response would reasonably have been appropriate.

Do not create an empathy opportunity for an ordinary factual question or a
normal request where there is no meaningful emotional/problem situation.

STEP 2 - EVALUATE THE AGENT'S RESPONSE
For every genuine empathy opportunity, inspect the agent's response to that
customer statement and decide whether the agent demonstrated empathy.

Empathy should be based on the meaning and context of the response, not on
whether the agent copied one of the example phrases below word-for-word.
Semantically equivalent language should count.

An agent can demonstrate empathy by acknowledging or validating the
customer's feelings, recognizing the inconvenience/problem, apologizing
appropriately when relevant, showing understanding, or expressing a clear
supportive intention to help in response to the customer's situation.

Do not mark PASS merely because the agent was polite, professional, or
continued helping. There should be evidence that the response addressed the
customer's emotional or difficult situation.

{EMPATHY_EXAMPLES}

EVIDENCE REQUIREMENT:
For EVERY genuine empathy opportunity, the evidence must contain BOTH:
1. The customer's actual statement that created the empathy opportunity.
2. The agent's actual corresponding response.

For a PASS, the pair should show the empathy opportunity and the agent's
empathetic response.

For a FAIL, the pair should show the empathy opportunity and the agent's
actual response that did not demonstrate empathy. Do NOT leave evidence
empty for a missed empathy opportunity.

For multiple empathy opportunities, return multiple customer/agent evidence
pairs.

Copy the evidence from the transcript as closely as possible. Do not invent,
paraphrase, or combine statements that were not said by the speakers.

Decision rules:

1. PASS when there is at least one genuine empathy opportunity and the agent
   demonstrates appropriate empathy in the relevant response(s).

2. FAIL when there is at least one genuine empathy opportunity and the agent
   does not demonstrate empathy for the relevant response(s).

3. NOT_APPLICABLE only when there is no genuine situation in the entire
   conversation where empathy was reasonably called for.

4. If there are several empathy opportunities, evaluate all of them before
   deciding PASS or FAIL.

5. Do not require the agent to use any exact sentence from the examples.

6. Prefer direct transcript evidence over assumptions.

7. The justification should be concise and explain why the agent passed,
   failed, or why the metric was not applicable.

TRANSCRIPT:

"""

    for chunk in transcript_chunks:
        prompt += f"""
Transcript chunk {chunk["chunk_id"]}:
{chunk["text"]}
"""

    return prompt


def evaluate_metric(
    llm,
    metric: dict,
    retrieved_chunks: list
) -> dict:

    logger.info(
        "Evaluating metric: %s",
        metric["name"]
    )

    if is_empathy_metric(metric):

        prompt = create_empathy_prompt(
            metric,
            retrieved_chunks
        )

        structured_llm = llm.with_structured_output(
            EmpathyEvaluationResult
        )

    else:

        prompt = create_prompt(
            metric,
            retrieved_chunks
        )

        structured_llm = llm.with_structured_output(
            EvaluationResult
        )

    response = structured_llm.invoke(
        prompt
    )

    return response.model_dump()


def calculate_score(
    metric: dict,
    evaluation: dict
) -> float:

    if evaluation["result"] == "PASS":

        return metric["weight"]

    return 0.0


def save_results(
    results: dict,
    recording_name: str
) -> str:

    results_path = Path(
        results_directory
    )

    results_path.mkdir(
        parents=True,
        exist_ok=True
    )

    output_file = (
        results_path /
        f"{recording_name}_analysis.json"
    )

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            results,
            file,
            indent=4
        )

    logger.info(
        "Results saved to: %s",
        output_file
    )

    return str(output_file)


def evaluate_transcript(
    transcript_path: str
) -> str:

    if not Path(transcript_path).exists():

        logger.error(
            "Transcript file not found: %s",
            transcript_path
        )

        return ""

    recording_name = Path(
        transcript_path
    ).stem

    metrics = load_metrics()

    index, metadata = load_transcript_index(
        transcript_path
    )

    model = load_embedding_model()

    metric_results = []

    total_score = 0.0

    maximum_score = 0.0

    fatal_metric_failed = False

    # Check once whether the runtime metric configuration contains
    # the special Empathy metric.
    empathy_metric_present = any(
        is_empathy_metric(metric)
        for metric in metrics
    )

    # Build the full transcript representation only when Empathy
    # is actually configured. Normal metrics continue using the
    # existing top-2 semantic retrieval flow.
    all_transcript_chunks = (
        get_all_transcript_chunks(metadata)
        if empathy_metric_present
        else []
    )

    for metric in metrics:

        logger.info(
            "Processing metric: %s",
            metric["name"]
        )

        if is_empathy_metric(metric):

            logger.info(
                "Empathy metric detected. Passing all transcript chunks to LLM."
            )

            retrieved_chunks = all_transcript_chunks

        else:

            criteria = metric["criteria"][0]

            retrieved_chunks = search_chunks(
                model,
                index,
                metadata,
                criteria
            )

        evaluation = evaluate_metric(
            llm,
            metric,
            retrieved_chunks
        )

        score = calculate_score(
            metric,
            evaluation
        )

        if (
            metric["metric_type"].lower() == "fatal"
            and
            evaluation["result"].lower() == "fail"
        ):

            fatal_metric_failed = True

        total_score += score

        maximum_score += metric["weight"]

        metric_results.append(
            {
                "name": metric["name"],
                "metric_type": metric["metric_type"],
                "weight": metric["weight"],
                "result": evaluation["result"],
                "score": score,
                "justification": evaluation["justification"],
                "evidence": evaluation["evidence"],
                "retrieved_chunks": retrieved_chunks
            }
        )

    if fatal_metric_failed:

        total_score = 0

    else:

        if maximum_score > 0:

            total_score = (
                total_score /
                maximum_score
            ) * 100

        else:

            total_score = 0

    results = {
        "recording": recording_name,
        "metrics": metric_results,
        "total_score": total_score,
        "maximum_score": maximum_score
    }

    result_path = save_results(
        results,
        recording_name
    )

    return result_path


def main() -> None:

    transcript_path = input(
        "Enter transcript path: "
    ).strip()

    result_path = evaluate_transcript(
        transcript_path
    )

    if result_path:

        print(
            f"\nAnalysis saved to: {result_path}"
        )


if __name__ == "__main__":

    main()
