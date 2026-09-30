"""Configuration-only checks; do not run training, evaluation, or authentication."""

import ast
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import pytest
from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "examples/deepcoder"
TEMPLATE = ROOT / ".env.example"
LAUNCHERS = sorted(EXAMPLES.rglob("*.sh"))
PYTHON_ENTRIES = sorted(EXAMPLES.glob("*.py"))


@pytest.fixture
def configured_checkout(tmp_path):
    checkout = tmp_path / "checkout with spaces"
    checkout.mkdir(parents=True)
    values = dict(dotenv_values(TEMPLATE))
    paths = {
        "DATA_ROOT": str(tmp_path / "shared data"),
        "CHECKPOINT_ROOT": str(tmp_path / "shared checkpoints"),
        "DEEPCODER_SWE_V4_1_TRAIN_FILE": str(tmp_path / "shared data/train.parquet"),
        "DEEPCODER_SWE_V4_1_VAL_FILE": str(tmp_path / "shared data/val.parquet"),
        "SFT_N10K_T1K_MODEL_PATH": str(tmp_path / "models/10k 1k/checkpoint"),
        "SFT_N8K_T3K_MODEL_PATH": str(tmp_path / "models/8k 3k/checkpoint"),
    }
    values.update(paths)
    for key, value in values.items():
        if value == "xxxxx":
            values[key] = "test-" + key.lower()
    env_file = checkout / ".env"
    env_file.write_text("\n".join(f"{key}={json.dumps(value)}" for key, value in values.items()) + "\n")
    environment = {key: value for key, value in os.environ.items() if key not in values and key not in {"RLLM_REPO_ROOT", "SLURM_SUBMIT_DIR", "WANDB_RUN_ID", "PYTHON_DOTENV_DISABLED"}}
    environment["PATH"] = str(Path(sys.executable).parent) + os.pathsep + environment.get("PATH", "/usr/bin:/bin")
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    return checkout, values, environment


@pytest.mark.parametrize("script", LAUNCHERS, ids=lambda p: str(p.relative_to(EXAMPLES)))
def test_shell_configuration_is_loaded_before_any_work(script, configured_checkout):
    checkout, shared, environment = configured_checkout
    text = script.read_text()
    assert "# === Runtime setup ===" in text
    prefix = text.split("# === Runtime setup ===", 1)[0]
    assert "wandb login" not in prefix
    assert 'load_dotenv(".env", override=False)' in prefix
    assert prefix.index("load_dotenv") < prefix.index("# === Script-specific configuration ===")
    environment["WANDB_ENTITY"] = "caller-team"
    environment["WANDB_RUN_ID"] = "caller-run"
    environment["API_KEY"] = ""  # Even an explicitly empty external value wins.
    local_names = ["train_files", "val_files", "model_path", "output_dir", "INPUT_PATH", "OUTPUT_PATH", "exp_name", "MONITOR_MODEL"]
    args = " ".join('"${' + name + '-}"' for name in local_names)
    capture = '\npython3 - ' + args + " <<'CAPTURE'\nimport json, os, sys\nprint(json.dumps({'local': sys.argv[1:], 'env': dict(os.environ)}))\nCAPTURE\n"
    target = checkout / script.relative_to(ROOT)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(prefix + capture)
    # Only the bootstrap and assignments are executed; the runtime block is absent.
    result = subprocess.run(["bash", str(target)], cwd=checkout, env=environment, capture_output=True, text=True, check=True, timeout=20)
    payload = json.loads(result.stdout)
    local = dict(zip(local_names, payload["local"]))
    env = payload["env"]
    assert env["WANDB_ENTITY"] == "caller-team"
    assert env["WANDB_RUN_ID"] == "caller-run"
    assert env["API_KEY"] == ""
    assert env["proj_name"] == "catch_rl"
    assert env["TRAIN_LARK_SAMPLES"] == env["VAL_LARK_SAMPLES"] == "0"
    if "python3 -m examples.deepcoder.train_deepcoder" in text:
        assert local["train_files"] == shared["DEEPCODER_SWE_V4_1_TRAIN_FILE"]
        assert local["val_files"] == shared["DEEPCODER_SWE_V4_1_VAL_FILE"]
        assert local["output_dir"] == str(Path(shared["CHECKPOINT_ROOT"]) / "catch_rl" / local["exp_name"])
        if 'model_path="$SFT_N10K_T1K_MODEL_PATH"' in text:
            assert local["model_path"] == shared["SFT_N10K_T1K_MODEL_PATH"]
        elif 'model_path="$SFT_N8K_T3K_MODEL_PATH"' in text:
            assert local["model_path"] == shared["SFT_N8K_T3K_MODEL_PATH"]
        else:
            assert local["model_path"].startswith(shared["CHECKPOINT_ROOT"] + "/")
    else:
        assert local["INPUT_PATH"] and local["OUTPUT_PATH"]
        assert env["API_MODEL"] == shared["API_MODEL"]


@pytest.mark.parametrize("entry", PYTHON_ENTRIES, ids=lambda p: p.name)
def test_direct_python_bootstrap_precedes_dependency_imports(entry, configured_checkout):
    checkout, _, environment = configured_checkout
    environment.update({"WANDB_ENTITY": "caller-team", "API_KEY": ""})
    tree = ast.parse(entry.read_text())
    loader_index = next(i for i, node in enumerate(tree.body) if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and node.value.func.id == "load_dotenv")
    prefix = ast.Module(body=tree.body[:loader_index + 1], type_ignores=[])
    for node in prefix.body:
        if isinstance(node, ast.Import):
            assert all(alias.name in {"os", "sys"} for alias in node.names)
        if isinstance(node, ast.ImportFrom):
            assert node.module in {"__future__", "pathlib", "dotenv"}
    source = ast.unparse(prefix)
    code = "__file__ = " + repr(str(checkout / entry.relative_to(ROOT))) + "\n" + source.replace("from __future__ import annotations\n", "")
    code += "\nimport json, os\nprint(json.dumps({'entity':os.environ.get('WANDB_ENTITY'),'key':os.environ.get('API_KEY'),'project':os.environ.get('proj_name')}))\n"
    result = subprocess.run([sys.executable, "-c", code], cwd=checkout, env=environment, capture_output=True, text=True, check=True, timeout=20)
    assert json.loads(result.stdout) == {"entity": "caller-team", "key": "", "project": "catch_rl"}


def test_bash_interpolation_and_quoting_match_python_dotenv(configured_checkout):
    checkout, _, environment = configured_checkout
    marker = checkout / "must-not-exist"
    (checkout / ".env").write_text(
        'DATA_ROOT="file data"\n'
        'DERIVED_PATH="${DATA_ROOT}/nested path"\n'
        'API_KEY="file-key"\n'
        f"LITERAL_TEXT='$(touch {marker}) # not shell code'\n"
    )
    environment.update({"DATA_ROOT": "caller data", "API_KEY": ""})
    prefix = LAUNCHERS[0].read_text().split("# === Script-specific configuration ===", 1)[0]
    capture = "\npython3 -c 'import json,os; print(json.dumps({k:os.environ[k] for k in [\"DATA_ROOT\",\"DERIVED_PATH\",\"API_KEY\",\"LITERAL_TEXT\"]}))'\n"
    result = subprocess.run(["bash", "-c", prefix + capture], cwd=checkout, env=environment, capture_output=True, text=True, check=True, timeout=20)
    data = json.loads(result.stdout)
    assert data["DATA_ROOT"] == "caller data"
    assert data["DERIVED_PATH"] == "caller data/nested path"
    assert data["API_KEY"] == ""
    assert data["LITERAL_TEXT"].startswith("$(touch ")
    assert not marker.exists()


def test_environment_loading_keeps_credentials_out_of_trace_output(configured_checkout):
    checkout, _, environment = configured_checkout
    (checkout / ".env").write_text('API_KEY="test-only-secret-value"\n')
    prefix = LAUNCHERS[0].read_text().split("# === Script-specific configuration ===", 1)[0]
    command = 'set -ax\n' + prefix + '\ncase "$-" in *a*) ;; *) exit 2 ;; esac\n'
    result = subprocess.run(["bash", "-c", command], cwd=checkout, env=environment, capture_output=True, text=True, check=True, timeout=20)
    assert "test-only-secret-value" not in result.stderr


def test_spooled_script_reads_env_from_submission_directory(configured_checkout):
    checkout, _, environment = configured_checkout
    original = next(p for p in LAUNCHERS if "#SBATCH" in p.read_text())
    prefix = original.read_text().split("# === Runtime setup ===", 1)[0]
    spool = checkout.parent / "slurm_script"
    spool.write_text(prefix + '\nprintf "%s" "$proj_name"\n')
    environment["SLURM_SUBMIT_DIR"] = str(checkout)
    result = subprocess.run(["bash", str(spool)], cwd=checkout.parent, env=environment, capture_output=True, text=True, check=True, timeout=20)
    assert result.stdout == "catch_rl"


@pytest.mark.parametrize("entry_name", ["prepare_deepcoder_swe_data.py", "prepare_skywork_swe_data.py"])
def test_conversion_parser_reads_shared_defaults_without_running_conversion(entry_name, configured_checkout, monkeypatch):
    import argparse

    checkout, shared, _ = configured_checkout
    for key in ["API_MODEL", "API_BASE_URL", "API_KEY"]:
        monkeypatch.setenv(key, shared[key])
    tree = ast.parse((EXAMPLES / entry_name).read_text())
    parser = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "parse_args")
    namespace = {"argparse": argparse, "os": os, "Path": Path, "DEFAULT_INPUT_PATH": Path("input.parquet"), "DEFAULT_OUTPUT_PATH": Path("output.parquet")}
    exec(compile(ast.Module(body=[parser], type_ignores=[]), entry_name, "exec"), namespace)
    monkeypatch.setattr(sys, "argv", [entry_name])
    args = namespace["parse_args"]()
    assert args.model == shared["API_MODEL"]
    assert args.base_url == shared["API_BASE_URL"]
    assert args.api_key == shared["API_KEY"]
    monkeypatch.setattr(sys, "argv", [entry_name, "--model", "cli-model", "--api-key", "cli-key"])
    args = namespace["parse_args"]()
    assert args.model == "cli-model" and args.api_key == "cli-key"


def test_template_contains_only_shared_settings_and_no_real_credentials():
    values = dotenv_values(TEMPLATE)
    assert values["proj_name"] == "catch_rl"
    forbidden = {"exp_name", "WANDB_RUN_ID", "CUDA_VISIBLE_DEVICES", "model_path", "output_dir", "INPUT_PATH", "OUTPUT_PATH", "MONITOR_MODEL", "SFT_N9K_T2K_MODEL_PATH", "SFT_N10_5K_T0_5K_MODEL_PATH"}
    assert not (set(values) & forbidden)
    for key, value in values.items():
        if key.endswith(("_KEY", "_SECRET", "_APP_ID", "_OPEN_ID", "_FILE", "_PATH", "_ROOT")) or key == "WANDB_ENTITY":
            assert value == "xxxxx", key
    assert not re.search(r"/home/fit/|/data/nvme\d+/|\bsk-[A-Za-z0-9_-]{16,}", TEMPLATE.read_text())


@pytest.mark.parametrize("script", sorted(EXAMPLES.rglob("*.sh")), ids=lambda p: str(p.relative_to(EXAMPLES)))
def test_shell_syntax(script):
    subprocess.run(["bash", "-n", str(script)], check=True)


def test_no_shared_helper_or_root_override_is_required():
    assert not (EXAMPLES / "common_env.sh").exists()
    for path in [*LAUNCHERS, *PYTHON_ENTRIES]:
        text = path.read_text()
        assert "common_env.sh" not in text, path
        assert "RLLM_REPO_ROOT" not in text, path


def test_user_guides_reference_current_scripts():
    guide = (EXAMPLES / "README.md").read_text()
    assert (ROOT / "docs/examples/deepcoder.md").read_text() == guide
    assert "repository root" in guide and ".env.example" in guide
    for stale in ["common_env.sh", "RLLM_REPO_ROOT", "train_deepcoder_16k.sh", "train_deepcoder_32k.sh", "Configuring the release scripts"]:
        assert stale not in guide
    paths = re.findall(r"examples/deepcoder/[A-Za-z0-9_./-]+\.(?:sh|py)", guide)
    assert paths
    assert all((ROOT / path).is_file() for path in paths)
