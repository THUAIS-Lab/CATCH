from rllm.agents.code_agent import CompetitionCodingAgent
from rllm.system_prompts import LCB_SYSTEM_MESSAGE_GENERIC


class TestCompetitionCodingAgent:
    def test_update_from_env_initial_prompt_frontloads_env_leakage(self):
        agent = CompetitionCodingAgent()

        observation = {
            "system_prompt": LCB_SYSTEM_MESSAGE_GENERIC,
            "env_leakage_prompt": "You are in a writable environment.",
            "question": "Solve the problem.",
            "format_prompt": "```python\n# solution.py\n```",
        }

        agent.update_from_env(observation, 0.0, False, {})

        assert len(agent.messages) == 1
        assert agent.messages[0]["role"] == "user"
        assert agent.messages[0]["content"] == (
            f"{LCB_SYSTEM_MESSAGE_GENERIC}\n\n"
            "You are in a writable environment.\n\n"
            "Solve the problem.\n\n"
            "```python\n# solution.py\n```"
        )

    def test_update_from_env_initial_prompt_respects_prompt_order(self):
        agent = CompetitionCodingAgent()

        observation = {
            "system_prompt": "Custom system prompt.",
            "env_leakage_prompt": "You are in a writable environment.",
            "question": "Solve the problem.",
            "format_prompt": "```python\n# solution.py\n```",
            "prompt_order": "question-system_prompt-env_leakage_prompt-format_prompt",
        }

        agent.update_from_env(observation, 0.0, False, {})

        assert agent.messages[0]["content"] == (
            "Solve the problem.\n\n"
            "Custom system prompt.\n\n"
            "You are in a writable environment.\n\n"
            "```python\n# solution.py\n```"
        )

    def test_update_from_env_initial_prompt_falls_back_to_question(self):
        agent = CompetitionCodingAgent()

        agent.update_from_env({"question": "Solve the problem."}, 0.0, False, {})

        assert agent.messages[0]["content"] == (
            f"{LCB_SYSTEM_MESSAGE_GENERIC}\n\n"
            "Solve the problem."
        )

    def test_update_from_env_initial_prompt_skips_empty_system_prompt(self):
        agent = CompetitionCodingAgent()

        agent.update_from_env(
            {
                "system_prompt": "",
                "question": "Solve the problem.",
            },
            0.0,
            False,
            {},
        )

        assert agent.messages[0]["content"] == "Solve the problem."

    def test_update_from_env_initial_prompt_skips_empty_sections_in_prompt_order(self):
        agent = CompetitionCodingAgent()

        agent.update_from_env(
            {
                "system_prompt": "",
                "env_leakage_prompt": "",
                "question": "Solve the problem.",
                "format_prompt": "```python\n# solution.py\n```",
                "prompt_order": "system_prompt-env_leakage_prompt-question-format_prompt",
            },
            0.0,
            False,
            {},
        )

        assert agent.messages[0]["content"] == (
            "Solve the problem.\n\n"
            "```python\n# solution.py\n```"
        )
