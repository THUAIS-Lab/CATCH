import os
import time
import uuid
from typing import Callable, Literal, Optional

from e2b.api import handle_api_exception
from e2b.api.client.api.templates import (
    post_templates,
    post_templates_template_id_builds_build_id,
)
from e2b.api.client.client import AuthenticatedClient
from e2b.api.client.models.error import Error
from e2b.connection_config import ConnectionConfig
from e2b.exceptions import BuildException
from e2b.template.logger import LogEntry
from e2b.template.types import BuildInfo

from e2b.api.client_sync import get_api_client
from e2b.api.client.types import UNSET, Unset, Union
from typing import Any


def get_build_status(
    client: AuthenticatedClient, template_id: str, build_id: str
):

    res = client.get_httpx_client().request(
        method="get",
        url=f"/templates/{template_id}/builds/{build_id}/status",
        params={},
    )

    if res.status_code >= 300:
        raise handle_api_exception(res, BuildException)

    return res.json()


def wait_for_build_finish(
    client: AuthenticatedClient,
    template_id: str,
    build_id: str,
    on_build_logs: Optional[Callable[[LogEntry], None]] = None,
    logs_refresh_frequency: float = 2,
):
    """
    Polls build status until completion and logs progress.

    :param client: Authenticated API client
    :param template_id: Template ID
    :param build_id: Build ID
    :param logs_refresh_frequency: Log refresh frequency (seconds, default: 0.2)
    """
    status: Literal["building", "waiting", "ready", "error"] = "building"

    logs_offset = 0
    while status in ["building", "waiting"]:
        build_status = get_build_status(client, template_id, build_id)

        status = build_status['status']
        logs = build_status['logs']

        for log in logs[logs_offset:]:
            if on_build_logs:
                on_build_logs(log)

        logs_offset = len(logs)

        if status == "ready":
            return

        elif status == "waiting":
            pass

        elif status == "error":
            error_message = (
                build_status['reason']['message']
                if 'reason' in build_status
                else "Build failed"
            )
            raise BuildException(error_message + f'\n{template_id=} {build_id=}')

        time.sleep(logs_refresh_frequency)

    raise BuildException("Unknown build error occurred.")

def get_build_info_if_exists(
    client: AuthenticatedClient,
    alias_without_uuid: str,
):
    res = client.get_httpx_client().request(
        method="get",
        url=f"/templates",
    )
    for build_info_dict in res.json():
        if any(alias_without_uuid in alias for alias in build_info_dict['aliases']):
            return BuildInfo(
                alias=build_info_dict['aliases'][0],
                template_id=build_info_dict['templateID'],
                build_id=build_info_dict['buildID'],
            )
    return None

def build_template(
    docker_image: str,
    alias: Optional[str] = None,
    cpu_count: int = 1,
    memory_mb: int = 1024,
    api_key: Optional[str] = None,
    domain: Optional[str] = None,
    wait_for_completion: bool = True,
    on_build_logs: Optional[Callable[[LogEntry], None]] = None,
    logs_refresh_frequency: float = 2,
    debug: bool = False,
) -> BuildInfo:
    """
    Build a simple template specifying a Docker image.

    :param docker_image: Docker image name (e.g. 'python:3')
    :param alias: Template alias (optional)
    :param cpu_count: Number of CPU cores (default: 2)
    :param memory_mb: Amount of memory in MB (default: 1024)
    :param api_key: E2B API key (optional, will read from environment variable if not provided)
    :param domain: E2B API domain (optional, will read from environment variable if not provided)
    :param wait_for_completion: Whether to wait for the build to complete (default: False)
    :param on_build_logs: Log callback function (optional)
    :param logs_refresh_frequency: Log refresh frequency (seconds, default: 0.2)
    :return: BuildInfo containing template_id and build_id

    Example
    ```python
    from build_template_v1 import build_template
    from e2b.template.logger import default_build_logger

    # Do not wait for build completion (default)
    build_info = build_template(
        docker_image='python:3',
        alias='my-python-template',
        cpu_count=2,
        memory_mb=1024
    )
    print(f"Template ID: {build_info.template_id}")
    print(f"Build ID: {build_info.build_id}")

    # Wait for build completion and log progress
    build_info = build_template(
        docker_image='python:3',
        alias='my-python-template',
        cpu_count=2,
        memory_mb=1024,
        wait_for_completion=True,
        on_build_logs=default_build_logger(min_level='info')
    )
    ```
    """
    # set default values
    domain = domain or os.environ.get("E2B_DOMAIN", "e2b.app")
    config = ConnectionConfig(
        domain=domain, api_key=api_key or os.environ.get("E2B_API_KEY")
    )
    api_client = get_api_client(
        config,
        require_api_key=True,
        require_access_token=False,
    )

    _alias = alias or f"{docker_image.replace(':', '-').replace('/', '-')}"

    build_info = get_build_info_if_exists(api_client, _alias)

    if  build_info is not None:
        return build_info
    else:
        # call POST /templates to create template
        res = post_templates._build_response(
            client=api_client,
            response=api_client.get_httpx_client().request(
                method="post",
                url="/templates",
                json={
                    "alias": _alias,
                    "cpuCount": cpu_count,
                    "memoryMB": memory_mb,
                },
                headers={"Content-Type": "application/json"},
            )
        )

    if res.status_code >= 300:
        if res.status_code == 403: # there has a build record with the same alias before, but the template is not ready
            if debug:
                print(f"There has a build record with the same alias before, but the template is not ready. {_alias=}")
            return build_template(
                docker_image=docker_image,
                alias=_alias+f'-{uuid.uuid4()}', # add a unique suffix to the alias to avoid conflict
                cpu_count=cpu_count,
                memory_mb=memory_mb,
                api_key=api_key,
                domain=domain,
                wait_for_completion=wait_for_completion,
            )
        raise handle_api_exception(res, BuildException)

    if isinstance(res.parsed, Error):
        raise BuildException(f"API error: {res.parsed.message}")

    if res.parsed is None:
        raise BuildException("Failed to create template")

    template_legacy = res.parsed
    template_id = template_legacy.template_id
    build_id = template_legacy.build_id

    # trigger build
    if debug:
        print(f"Triggering build. {template_id=} {build_id=}")
    build_res = post_templates_template_id_builds_build_id._build_response(
        client=api_client,
        response=api_client.get_httpx_client().request(
            method="post",
            url=f"/templates/{template_id}/builds/{build_id}",
            json={
                "registry": {
                    "imageName": docker_image,
                }
            },
            headers={"Content-Type": "application/json"},
        )
    )

    if build_res.status_code >= 300:
        raise handle_api_exception(build_res, BuildException)

    if isinstance(build_res.parsed, Error):
        raise BuildException(f"API error: {build_res.parsed.message}")

    build_info = BuildInfo(
        alias=alias or f"{docker_image.replace(':', '-').replace('/', '-')}",
        template_id=template_id,
        build_id=build_id,
    )

    # if wait for completion, poll status and log progress
    if wait_for_completion:
        wait_for_build_finish(
            api_client,
            template_id,
            build_id,
            on_build_logs=on_build_logs,
            logs_refresh_frequency=logs_refresh_frequency,
        )

    return build_info
