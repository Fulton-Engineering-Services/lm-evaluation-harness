import re

import evaluate as hf_evaluate


try:
    compute_ = hf_evaluate.load("code_eval")
    test_cases = ["assert add(2, 3)==5"]
    candidates = [["def add(a,b): return a*b"]]
    results = compute_.compute(references=test_cases, predictions=candidates, k=[1])
except Exception as e:
    raise e


def pass_at_k(references: list[str], predictions: list[list[str]], k: list[int] = None):
    global compute_
    assert k is not None
    if isinstance(k, int):
        k = [k]
    res = compute_.compute(
        references=references,
        predictions=predictions,
        k=k,
    )
    return res[0]


def build_predictions(resps: list[list[str]], docs: list[dict]) -> list[list[str]]:
    return [[doc["prompt"] + r for r in resp] for resp, doc in zip(resps, docs)]


def build_predictions_instruct(
    resps: list[list[str]], docs: list[dict]
) -> list[list[str]]:
    """Assemble the scored prediction for instruct-style HumanEval prompts.

    ``code_eval`` scores ``candidate + "\n" + test_case``, and the task's
    ``doc_to_target`` is that program. The candidate must therefore be a runnable
    program in its own right -- concatenating the raw completion onto the prompt
    only works when the model merely *continues* the fence that ``gen_prefix``
    left open.

    Models whose chat template closes that fence before generating instead open a
    new one, so the completion carries its own fenced block. Upstream truncated
    at the FIRST fence, which for those models sits at index 0: the prediction
    collapsed to the bare prompt and every sample scored zero. Extract the
    completion's fenced block when it has one, and otherwise use the prompt plus
    the completion verbatim (the fence-continuing shape upstream assumed).
    """
    return [
        [
            _prediction(r, doc["prompt"])
            for r in resp
        ]
        for resp, doc in zip(resps, docs)
    ]


_FENCED = re.compile(r"```(?:\w+)?\n?(.*?)\n?```", re.DOTALL)


def _prediction(completion: str, prompt: str) -> str:
    """Return a runnable program for one completion.

    A completion that carries a fenced block is extracted to its bare code (the
    block is self-contained: HumanEval instruct prompts ask for the whole
    function). A completion with no fence at all is appended to the prompt, which
    is the shape upstream's ``gen_prefix`` expects.
    """
    blocks = _FENCED.findall(completion)
    if blocks:
        return blocks[-1]
    return prompt + completion
