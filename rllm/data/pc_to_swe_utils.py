from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class ExtractedDeepCoderInputTail:
    question_before_input_section: str
    input_and_following_text: str | None


SECTION_HEADING_TEMPLATE = r"^[ \t>#*-]*(?:{name})[ \t:.-]*$"
INPUT_SECTION_RE = re.compile(
    SECTION_HEADING_TEMPLATE.format(
        name=(
            r"(?:Inputs?(?:\s+(?:Format|Specification|Description|Data))?"
            r"|inputs?(?:\s+(?:format|specification|description|data))?"
            r"|INPUTS?(?:\s+(?:FORMAT|SPECIFICATION|DESCRIPTION|DATA))?"
            r"|Входные\ данные)"
        )
    ),
    flags=re.IGNORECASE | re.MULTILINE,
)
CONSTRAINTS_SECTION_RE = re.compile(
    SECTION_HEADING_TEMPLATE.format(
        name=(
            r"(?:Constraints?(?:\s+(?:Format|Specification|Description))?"
            r"|constraints?(?:\s+(?:format|specification|description))?"
            r"|CONSTRAINTS?(?:\s+(?:FORMAT|SPECIFICATION|DESCRIPTION))?"
            r"|Ограничения)"
        )
    ),
    flags=re.IGNORECASE | re.MULTILINE,
)
STDIO_INSTRUCTION_SECTION_RE = re.compile(
    r"(?im)^(?:the input .*stdin.*stdout.*|input .*stdin.*stdout.*)$"
)
EXAMPLES_SECTION_RE = re.compile(
    r"(?im)^[ \t>#*-]*\*{0,2}(?:Examples?)\*{0,2}[ \t:.\-]*\*{0,2}$"
)
WRITE_SOLUTION_SECTION_RE = re.compile(
    r"(?im)^[ \t>#*-]*write your solution by modifying this code:[ \t]*$"
)
OUTPUT_SECTION_RE = re.compile(
    SECTION_HEADING_TEMPLATE.format(
        name=(
            r"(?:Outputs?(?:\s+(?:Format|Specification|Description|Data))?"
            r"|outputs?(?:\s+(?:format|specification|description|data))?"
            r"|OUTPUTS?(?:\s+(?:FORMAT|SPECIFICATION|DESCRIPTION|DATA))?"
            r"|Выходные\ данные)"
        )
    ),
    flags=re.IGNORECASE | re.MULTILINE,
)


def _collapse_blank_lines(text: str) -> str:
    return re.sub(r"\n{3,}", "\n\n", text.strip())


def extract_deepcoder_input_and_following_sections(
    question: str,
    *,
    task_kind: str | None = None,
) -> ExtractedDeepCoderInputTail:
    """Extract the Input section and all following text from a DeepCoder question.

    This is intentionally conservative. If an explicit Constraints heading appears
    before the Input heading, extraction starts from Constraints; otherwise it
    starts from Input. If neither heading exists, the function falls back to a
    trailing stdin/stdout instruction sentence, then to an explicit Examples
    heading, and finally for `function_call` tasks to the starter-code preamble
    line. If none of these cues exists, it returns the original question
    unchanged and leaves `input_and_following_text` as `None`.
    """
    normalized_question = question.replace("\r\n", "\n").strip()
    input_match = INPUT_SECTION_RE.search(normalized_question)
    constraints_match = CONSTRAINTS_SECTION_RE.search(normalized_question)
    if constraints_match is not None and (
        input_match is None or constraints_match.start() < input_match.start()
    ):
        start_match = constraints_match
    else:
        start_match = input_match
    if start_match is None:
        stdio_instruction_match = STDIO_INSTRUCTION_SECTION_RE.search(normalized_question)
        if stdio_instruction_match is not None:
            return ExtractedDeepCoderInputTail(
                question_before_input_section=_collapse_blank_lines(
                    normalized_question[: stdio_instruction_match.start()]
                ),
                input_and_following_text=_collapse_blank_lines(
                    normalized_question[stdio_instruction_match.start() :]
                ),
            )

        examples_match = EXAMPLES_SECTION_RE.search(normalized_question)
        if examples_match is not None:
            return ExtractedDeepCoderInputTail(
                question_before_input_section=_collapse_blank_lines(
                    normalized_question[: examples_match.start()]
                ),
                input_and_following_text=_collapse_blank_lines(
                    normalized_question[examples_match.start() :]
                ),
            )

        if task_kind == "function_call":
            write_solution_match = WRITE_SOLUTION_SECTION_RE.search(normalized_question)
            if write_solution_match is not None:
                return ExtractedDeepCoderInputTail(
                    question_before_input_section=_collapse_blank_lines(
                        normalized_question[: write_solution_match.start()]
                    ),
                    input_and_following_text=_collapse_blank_lines(
                        normalized_question[write_solution_match.start() :]
                    ),
                )

        return ExtractedDeepCoderInputTail(
            question_before_input_section=normalized_question,
            input_and_following_text=None,
        )

    # Some kata statements contain small example tables with "Input" / "Output"
    # column headers. If the text between those headings is empty, do not treat the
    # table as a real contest I/O section.
    output_match = OUTPUT_SECTION_RE.search(normalized_question, start_match.end())
    if output_match is not None:
        between_input_and_output = _collapse_blank_lines(
            normalized_question[start_match.end() : output_match.start()]
        )
        if not between_input_and_output:
            return ExtractedDeepCoderInputTail(
                question_before_input_section=normalized_question,
                input_and_following_text=None,
            )

    question_before_input_section = _collapse_blank_lines(
        normalized_question[: start_match.start()]
    )
    input_and_following_text = _collapse_blank_lines(
        normalized_question[start_match.start() :]
    )
    if not input_and_following_text:
        return ExtractedDeepCoderInputTail(
            question_before_input_section=normalized_question,
            input_and_following_text=None,
        )

    return ExtractedDeepCoderInputTail(
        question_before_input_section=question_before_input_section,
        input_and_following_text=input_and_following_text,
    )
