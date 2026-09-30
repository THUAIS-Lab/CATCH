import os
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
from threading import Thread, Lock
from typing import Dict, Optional, List, Set
from verl import DataProto

from e2b.api.client.api.templates.delete_templates_template_id import sync as delete_template_sync
from e2b.api import ApiClient
from e2b.connection_config import ConnectionConfig
import uuid
from . import build_template_v1
from rllm.utils.visualization import colorful_print

class TemplatePool:
    """
    Manages the construction and release of E2B templates, enabling pipelined parallelism.
    
    Constraints:
    - The combined number of currently building and completed templates cannot exceed max_templates (default is 2 * batch_size)
    - When the limit is reached, it will wait for a built template to be used and released before proceeding
    """
    
    def __init__(self, max_templates: int, max_build_workers: int, backend: str, delete_template=False, debug=False):
        """
        Args:
            max_templates: Maximum number of templates being built or already built at the same time
            api_key: E2B API key. If None, it will be obtained from the environment variable
        """
        try:
            from e2b import Template, default_build_logger
            self.Template = Template
            self.default_build_logger = default_build_logger
        except ImportError:
            raise ImportError("e2b package is not installed. Please install it with `pip install e2b`.")
        
        self.max_templates = max_templates
        self.max_build_workers = max_build_workers
        self.backend = backend
        assert self.backend in ["e2b", "ppio"], f"Unsupported backend: {self.backend}, must be one of ['e2b','ppio']"
        self.delete_template = delete_template
        
        # docker_image -> template_id (for completed builds)
        self._image_to_template_id: Dict[str, str] = {}
        # docker_image -> Future (for ongoing builds)
        self._image_to_future: Dict[str, Future] = {}
        # Set of images marked for cleanup
        self._marked_to_clean: Set[str] = set()
        # Lock for thread-safe operations
        self._lock = Lock()
        # Thread pool for building templates
        self._build_executor = ThreadPoolExecutor(max_workers=max_build_workers, thread_name_prefix="e2b_build")
        # Thread pool for deleting templates
        self._cleanup_executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="e2b_cleanup")
        self.debug = debug
    
    def _build_template_with_retry(self, image: str) -> str:
        """
        Build a template from a docker image with retry logic.
        
        Args:
            image: Docker image name
            
        Returns:
            template_id: The ID of the built template
        """
        attempt = 0
        max_retries = 3
        backoff = 60
        while True:
            try:
                alias = image.replace(':', '-').replace('/', '-')
                if self.backend == "e2b":
                    template = self.Template().from_image(f'hub.1panel.dev/{image}').set_user('root')
                    build_info = self.Template.build(
                        alias=alias,
                        template=template,
                        cpu_count=1,
                        memory_mb=1024,
                        on_build_logs=None
                    )
                elif self.backend == "ppio":
                    build_info = build_template_v1.build_template(
                        alias=alias,
                        docker_image=f'hub.1panel.dev/{image}',
                        cpu_count=1,
                        memory_mb=1024,
                        on_build_logs=None
                    )
                else:
                    raise ValueError(f"Unsupported backend: {self.backend}, must be one of ['e2b','ppio']")

                if self.debug:
                    print(f"[build_template_with_retry] {image=} {build_info=}")

                template_id = build_info.template_id
                return template_id

            except Exception as e:
                if 'limit' in e.__str__():
                    if self.debug:
                        print(f"[build_template_with_retry] Rate limit exceeded for {image}. Retrying in {backoff}s...")
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 20)
                    continue
                elif attempt < max_retries - 1:
                    if self.debug:
                        print(f"[build_template_with_retry] Failed to build template for {image} (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {backoff}s...")
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 300)
                    attempt += 1
                    continue
                else:
                    print(f"[build_template_with_retry] Failed to build template for {image} after {max_retries} attempts: {e}")
                    raise
    
    def build_template_async(self, batch: DataProto):
        """
        Asynchronously submit template build tasks for a batch.
        
        Args:
            batch: DataProto containing docker images in extra_info
        """
        env_args = batch.non_tensor_batch['extra_info'].tolist()
        docker_images = [entry['docker_image'] for entry in env_args]
        
        with self._lock:
            for image in docker_images:
                # Skip if already building or built
                if image in self._image_to_future or image in self._image_to_template_id:
                    continue

                # Submit build task
                future = self._build_executor.submit(self._build_template_with_retry, image)
                self._image_to_future[image] = future

    def get_template_ids(self, batch: DataProto) -> DataProto:
        """
        Get template IDs for a batch, waiting for builds to complete if necessary.
        
        Args:
            batch: DataProto containing docker images in extra_info
            
        Returns:
            DataProto with template_ids in non_tensor_batch
        """
        env_args = batch.non_tensor_batch['extra_info'].tolist()
        docker_images = [entry['docker_image'] for entry in env_args]
        template_ids = []
        
        for image in docker_images:
            # Retry loop: keep retrying until build succeeds
            future: Optional[Future] = None
            with self._lock:
                # Check if already built
                if image in self._image_to_template_id:
                    if self.debug:
                        print(f"[get_teimpalte_ids] Image {image} built successfully with template id {template_id}")
                        pass
                    template_ids.append(self._image_to_template_id[image])
                    continue
                
                # Get or create future
                if image in self._image_to_future:
                    future = self._image_to_future[image]
                else:
                    # No future exists, create a new one
                    if self.debug:
                        print(f"[get_teimpalte_ids] No future exists for {image}, creating a new one.")
                    future = self._build_executor.submit(self._build_template_with_retry, image)
                    self._image_to_future[image] = future

            # Wait for build to complete (outside lock to avoid deadlock)
            try:
                if self.debug:
                    print(f"[get_teimpalte_ids] Image {image} waiting for build to complete.")
                template_id = future.result()
                # print(f"[get_teimpalte_ids] Image {image} built successfully with template id {template_id}")
                with self._lock:
                    self._image_to_template_id[image] = template_id
                template_ids.append(template_id)
                continue
            except Exception as e:
                colorful_print(f"[get_template_ids] Error building template for {image}: {e}. Skipping...", "red")
                # Remove failed future
                with self._lock:
                    self._image_to_future.pop(image)
                    template_ids.append(None)

        return DataProto.from_dict(non_tensors={
            'template_id': template_ids
        })

    def is_build_queue_full(self) -> bool:
        """
        Check if the build queue is full (reached max_templates limit).
        
        Returns:
            bool: True if the queue is full, False otherwise
        """
        with self._lock:
            return len(self._image_to_future) > self.max_templates

    def mark_batch_template_to_clean(self, batch: DataProto):
        """
        Mark templates in a batch for cleanup.
        
        Args:
            batch: DataProto containing docker images in extra_info
        """
        env_args = batch.non_tensor_batch['extra_info'].tolist()
        docker_images = [entry['docker_image'] for entry in env_args]
        
        with self._lock:
            marked = {image for image in docker_images if image in self._image_to_template_id or image in self._image_to_future}
            self._marked_to_clean.update(marked)

    def _delete_template_with_retry(self, template_id: str, image: str="Not Known"):
        """
        Delete a template with retry logic.
        
        Args:
            template_id: The ID of the template to delete
            image: The docker image name (for logging)
        """
        max_retries = 20
        backoff = 2  # seconds
        
        for attempt in range(max_retries):
            try:
                config = ConnectionConfig()
                client = ApiClient(config, require_api_key=True)
                delete_template_sync(template_id=template_id, client=client)
                return
            except Exception as e:
                if attempt < max_retries - 1:
                    if self.debug:
                        print(f"[delete_template_with_retry] Failed to delete template {template_id} for {image} (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {backoff}s...")
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 40)
                    continue
                else:
                    if self.debug:
                        print(f"[delete_template_with_retry] Failed to delete template {template_id} for {image} after {max_retries} attempts: {e}")
                    # Don't raise, just log the error

    def clean_marked_template_async(self):
        """
        Asynchronously delete templates that have been marked for cleanup.
        Each deletion is submitted as a separate task to enable parallel deletion.
        This runs in a separate thread pool to avoid blocking the main training loop.
        """
        with self._lock:
            images_to_clean = list(self._marked_to_clean)
            self._marked_to_clean.clear()
        
        if not images_to_clean:
            return
        
        def cleanup_single_template(image: str):
            """Clean up a single template."""
            with self._lock:
                if image not in self._image_to_template_id:
                    return
                template_id = self._image_to_template_id[image]
                # Remove from dict before deletion to avoid race conditions
                self._image_to_template_id.pop(image, None)
                self._image_to_future.pop(image, None)

            # Delete template (outside lock)
            if self.delete_template:
                self._delete_template_with_retry(template_id, image)
        
        # Submit each deletion as a separate task for parallel execution
        for image in images_to_clean:
            self._cleanup_executor.submit(cleanup_single_template, image)
