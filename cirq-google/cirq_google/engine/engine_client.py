# Copyright 2020 The Cirq Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import asyncio
import concurrent.futures
import datetime
import queue
import sys
import threading
import warnings
from collections.abc import AsyncIterable, Awaitable, Callable
from functools import cached_property
from typing import Any, Dict, TypeVar

import proto
from google.api_core.exceptions import GoogleAPICallError, NotFound
from google.protobuf import any_pb2, field_mask_pb2
from google.protobuf.timestamp_pb2 import Timestamp

from cirq import _compat
from cirq_google.cloud import quantum
from cirq_google.engine import stream_manager
from cirq_google.engine.asyncio_executor import AsyncioExecutor
from cirq_google.engine.processor_config import DeviceConfigRevision, Run, Snapshot

_M = TypeVar('_M', bound=proto.Message)
_R = TypeVar('_R')


def _fix_deprecated_allowlisted_users_args(
    args: tuple[Any, ...], kwargs: dict[str, Any]
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    kwargs['allowlisted_users'] = kwargs.pop('whitelisted_users')
    return args, kwargs


class EngineException(Exception):
    def __init__(self, message, cause=None):
        # Call the base class constructor with the parameters it needs
        super().__init__(message)
        self.__cause__ = cause


class EngineClient:
    """Client for the Quantum Engine API handling protos and gRPC client.

    This is the client for the Quantum Engine API that deals with the engine protos
    and the gRPC client but not cirq protos or objects. All users are likely better
    served by using the `Engine`, `EngineProgram`, `EngineJob`, `EngineProcessor`, and
    `Calibration` objects instead of using this directly.
    """

    def __init__(
        self,
        service_args: dict | None = None,
        verbose: bool | None = None,
        max_retry_delay_seconds: int = 3600,  # 1 hour
    ) -> None:
        """Constructs a client for the Quantum Engine API.

        Args:
            service_args: A dictionary of arguments that can be used to
                configure options on the underlying gRPC client.
            verbose: Suppresses stderr messages when set to False. Default is
                true.
            max_retry_delay_seconds: The maximum number of seconds to retry when
                a retryable error code is returned.
        """
        self.max_retry_delay_seconds = max_retry_delay_seconds
        self.verbose = verbose if verbose is not None else True
        self._service_args = service_args or {}
        self._loop_clients: Dict[int, quantum.QuantumEngineServiceAsyncClient] = {}

    def _create_grpc_client(self) -> quantum.QuantumEngineServiceAsyncClient:
        """Creates a fresh async grpc client."""
        # Suppress warnings about using Application Default Credentials.
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return quantum.QuantumEngineServiceAsyncClient(**self._service_args)

    def _get_loop_aware_client(self) -> quantum.QuantumEngineServiceAsyncClient:
        """Get or create a grpc client that binds to the current loop."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return self._create_grpc_client()

        loop_id = id(loop)
        if loop_id not in self._loop_clients:
            self._loop_clients[loop_id] = self._create_grpc_client()
        return self._loop_clients[loop_id]

    @property
    def grpc_client(self) -> quantum.QuantumEngineServiceAsyncClient:
        """Returns a grpc client compatible with the current event loop."""
        return self._get_loop_aware_client()

    @property
    def _stream_manager(self) -> stream_manager.StreamManager:
        return stream_manager.StreamManager(self.grpc_client)

    def _run_isolated_sync(
        self, method_name: str, request: proto.Message, is_list_operation: bool = False
    ) -> Any:
        """Executes an async gRPC method in a dedicated, isolated thread.

        Args:
            method_name: The name of the method on the generated client (e.g., 'list_quantum_jobs').
            request: The protobuf request object.
            is_list_operation: If True, iterates the response and returns a list.
                               If False, awaits and returns the single response object.
        """
        result_queue: queue.Queue[Any] = queue.Queue()

        def worker():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            client = self._create_grpc_client()

            async def do_work():
                try:
                    grpc_method = getattr(client, method_name)

                    if is_list_operation:
                        pager = await grpc_method(request)
                        data = [x async for x in pager]
                        result_queue.put(data)
                    else:
                        response = await grpc_method(request)
                        result_queue.put(response)

                except Exception as e:
                    result_queue.put(e)

            try:
                loop.run_until_complete(do_work())
            finally:
                loop.close()

        t = threading.Thread(target=worker, daemon=True)
        t.start()
        t.join()

        result = result_queue.get()
        if isinstance(result, Exception):
            if isinstance(result, GoogleAPICallError):
                raise EngineException(result.message, cause=result) from result
            raise result
        return result

    # --- Create Program ---

    def _build_create_program_request(self, project_id, program_id, code, description, labels):
        program_name = _program_name_from_ids(project_id, program_id) if program_id else ''
        program = quantum.QuantumProgram(name=program_name, code=code)
        if description:
            program.description = description
        if labels:
            program.labels.update(labels)

        return quantum.CreateQuantumProgramRequest(
            parent=_project_name(project_id), quantum_program=program
        )

    async def create_program_async(
        self,
        project_id: str,
        program_id: str | None,
        code: any_pb2.Any,
        description: str | None = None,
        labels: dict[str, str] | None = None,
    ) -> tuple[str, quantum.QuantumProgram]:
        """Creates a Quantum Engine program.

        Args:
            project_id: A project_id of the parent Google Cloud Project.
            program_id: Unique ID of the program within the parent project.
            code: Properly serialized program code.
            description: An optional description to set on the program.
            labels: Optional set of labels to set on the program.

        Returns:
            Tuple of created program id and program
        """
        request = self._build_create_program_request(
            project_id, program_id, code, description, labels
        )
        program = await self.grpc_client.create_quantum_program(request)
        return _ids_from_program_name(program.name)[1], program

    def create_program(
        self,
        project_id: str,
        program_id: str | None,
        code: any_pb2.Any,
        description: str | None = None,
        labels: dict[str, str] | None = None,
    ) -> tuple[str, quantum.QuantumProgram]:
        """Creates a Quantum Engine program. Prefer create_program_async()."""
        request = self._build_create_program_request(
            project_id, program_id, code, description, labels
        )
        program = self._run_isolated_sync('create_quantum_program', request)
        return _ids_from_program_name(program.name)[1], program

    # --- Get Program ---

    async def get_program_async(
        self, project_id: str, program_id: str, return_code: bool
    ) -> quantum.QuantumProgram:
        """Returns a previously created quantum program.

        Args:
            project_id: A project_id of the parent Google Cloud Project.
            program_id: Unique ID of the program within the parent project.
            return_code: If True returns the serialized program code.
        """
        request = quantum.GetQuantumProgramRequest(
            name=_program_name_from_ids(project_id, program_id), return_code=return_code
        )
        return await self.grpc_client.get_quantum_program(request)

    def get_program(
        self, project_id: str, program_id: str, return_code: bool
    ) -> quantum.QuantumProgram:
        """Returns a previously created quantum program. Prefer get_program_async()."""
        request = quantum.GetQuantumProgramRequest(
            name=_program_name_from_ids(project_id, program_id), return_code=return_code
        )
        return self._run_isolated_sync('get_quantum_program', request)

    # --- List Programs ---

    def _build_list_programs_request(self, project_id, created_before, created_after, has_labels):
        filters = []
        if created_after is not None:
            val = _date_or_time_to_filter_expr('created_after', created_after)
            filters.append(f"create_time >= {val}")
        if created_before is not None:
            val = _date_or_time_to_filter_expr('created_before', created_before)
            filters.append(f"create_time <= {val}")
        if has_labels is not None:
            for k, v in has_labels.items():
                filters.append(f"labels.{k}:{v}")
        return quantum.ListQuantumProgramsRequest(
            parent=_project_name(project_id), filter=" AND ".join(filters)
        )

    async def list_programs_async(
        self,
        project_id: str,
        created_before: datetime.datetime | datetime.date | None = None,
        created_after: datetime.datetime | datetime.date | None = None,
        has_labels: dict[str, str] | None = None,
    ):
        """Returns a list of previously executed quantum programs.

        Args:
            project_id: the id of the project
            created_after: retrieve programs that were created after this date
                or time.
            created_before: retrieve programs that were created after this date
                or time.
            has_labels: retrieve programs that have labels on them specified by
                this dict. If the value is set to `*`, filters having the label
                regardless of the label value will be filtered. For example, to
                query programs that have the shape label and have the color
                label with value red can be queried using

                {'color': 'red', 'shape':'*'}
        """
        request = self._build_list_programs_request(
            project_id, created_before, created_after, has_labels
        )

        pager = await self.grpc_client.list_quantum_programs(request)
        return [p async for p in pager]

    def list_programs(
        self,
        project_id: str,
        created_before: datetime.datetime | datetime.date | None = None,
        created_after: datetime.datetime | datetime.date | None = None,
        has_labels: dict[str, str] | None = None,
    ) -> list[quantum.QuantumProgram]:
        """Returns a list of previously executed quantum programs. Prefer list_programs_async()."""
        request = self._build_list_programs_request(
            project_id, created_before, created_after, has_labels
        )
        return self._run_isolated_sync('list_quantum_programs', request, is_list_operation=True)

    # --- Set Program Description ---

    def _build_set_program_description_request(self, project_id, program_id, description):
        program_resource_name = _program_name_from_ids(project_id, program_id)
        return quantum.UpdateQuantumProgramRequest(
            name=program_resource_name,
            quantum_program=quantum.QuantumProgram(
                name=program_resource_name, description=description
            ),
            update_mask=field_mask_pb2.FieldMask(paths=['description']),
        )

    async def set_program_description_async(
        self, project_id: str, program_id: str, description: str
    ) -> quantum.QuantumProgram:
        """Sets the description for a previously created quantum program.

        Args:
            project_id: A project_id of the parent Google Cloud Project.
            program_id: Unique ID of the program within the parent project.
            description: The new program description.

        Returns:
            The updated quantum program.
        """
        request = self._build_set_program_description_request(project_id, program_id, description)
        return await self.grpc_client.update_quantum_program(request)

    def set_program_description(
        self, project_id: str, program_id: str, description: str
    ) -> quantum.QuantumProgram:
        """Sets the description for a previously created quantum program.
        Prefer set_program_description_async()."""
        request = self._build_set_program_description_request(project_id, program_id, description)
        return self._run_isolated_sync('update_quantum_program', request)

    # --- Set Program Labels ---

    def _build_set_program_labels_request(self, project_id, program_id, labels, fingerprint):
        program_resource_name = _program_name_from_ids(project_id, program_id)
        return quantum.UpdateQuantumProgramRequest(
            name=program_resource_name,
            quantum_program=quantum.QuantumProgram(
                name=program_resource_name, labels=labels, label_fingerprint=fingerprint
            ),
            update_mask=field_mask_pb2.FieldMask(paths=['labels']),
        )

    async def _set_program_labels_async(
        self, project_id: str, program_id: str, labels: dict[str, str], fingerprint: str
    ) -> quantum.QuantumProgram:
        request = self._build_set_program_labels_request(
            project_id, program_id, labels, fingerprint
        )
        return await self.grpc_client.update_quantum_program(request)

    async def set_program_labels_async(
        self, project_id: str, program_id: str, labels: dict[str, str]
    ) -> quantum.QuantumProgram:
        """Sets (overwriting) the labels for a previously created quantum
        program.

        Args:
            project_id: A project_id of the parent Google Cloud Project.
            program_id: Unique ID of the program within the parent project.
            labels: The entire set of new program labels.

        Returns:
            The updated quantum program.
        """
        program = self.get_program(project_id, program_id, False)
        return await self._set_program_labels_async(
            project_id, program_id, labels, program.label_fingerprint
        )

    def _set_program_labels(
        self, project_id: str, program_id: str, labels: dict[str, str], fingerprint: str
    ) -> quantum.QuantumProgram:
        request = self._build_set_program_labels_request(
            project_id, program_id, labels, fingerprint
        )
        return self._run_isolated_sync('update_quantum_program', request)

    def set_program_labels(
        self, project_id: str, program_id: str, labels: dict[str, str]
    ) -> quantum.QuantumProgram:
        """Sets (overwriting) the labels for a previously created quantum
        program. Prefer set_program_labels_async()."""
        program = self.get_program(project_id, program_id, False)
        return self._set_program_labels(project_id, program_id, labels, program.label_fingerprint)

    # --- Add Program Labels ---

    async def add_program_labels_async(
        self, project_id: str, program_id: str, labels: dict[str, str]
    ) -> quantum.QuantumProgram:
        """Adds new labels to a previously created quantum program.

        Args:
            project_id: A project_id of the parent Google Cloud Project.
            program_id: Unique ID of the program within the parent project.
            labels: New labels to add to the existing program labels.

        Returns:
            The updated quantum program.
        """
        program = await self.get_program_async(project_id, program_id, False)
        old_labels = program.labels
        new_labels = dict(old_labels)
        new_labels.update(labels)
        if new_labels != old_labels:
            fingerprint = program.label_fingerprint
            return await self._set_program_labels_async(
                project_id, program_id, new_labels, fingerprint
            )
        return program

    def add_program_labels(
        self, project_id: str, program_id: str, labels: dict[str, str]
    ) -> quantum.QuantumProgram:
        """Adds new labels to a previously created quantum program.
        Prefer add_program_labels_async()."""
        program = self.get_program(project_id, program_id, False)
        old_labels = program.labels
        new_labels = dict(old_labels)
        new_labels.update(labels)
        if new_labels != old_labels:
            fingerprint = program.label_fingerprint
            return self._set_program_labels(project_id, program_id, new_labels, fingerprint)
        return program

    # --- Remove Program Labels ---

    async def remove_program_labels_async(
        self, project_id: str, program_id: str, label_keys: list[str]
    ) -> quantum.QuantumProgram:
        """Removes labels with given keys from the labels of a previously
        created quantum program.

        Args:
            project_id: A project_id of the parent Google Cloud Project.
            program_id: Unique ID of the program within the parent project.
            label_keys: Label keys to remove from the existing program labels.

        Returns:
            The updated quantum program.
        """
        program = await self.get_program_async(project_id, program_id, False)
        old_labels = program.labels
        new_labels = dict(old_labels)
        for key in label_keys:
            new_labels.pop(key, None)
        if new_labels != old_labels:
            fingerprint = program.label_fingerprint
            return await self._set_program_labels_async(
                project_id, program_id, new_labels, fingerprint
            )
        return program

    def remove_program_labels(
        self, project_id: str, program_id: str, label_keys: list[str]
    ) -> quantum.QuantumProgram:
        """Removes labels with given keys from the labels of a previously
        created quantum program. Prefer remove_program_labels_async()."""
        program = self.get_program(project_id, program_id, False)
        old_labels = program.labels
        new_labels = dict(old_labels)
        for key in label_keys:
            new_labels.pop(key, None)
        if new_labels != old_labels:
            fingerprint = program.label_fingerprint
            return self._set_program_labels(project_id, program_id, new_labels, fingerprint)
        return program

    # --- Delete Program ---

    async def delete_program_async(
        self, project_id: str, program_id: str, delete_jobs: bool = False
    ) -> None:
        """Deletes a previously created quantum program.

        Args:
            project_id: A project_id of the parent Google Cloud Project.
            program_id: Unique ID of the program within the parent project.
            delete_jobs: If True will delete all the program's jobs, other this
                will fail if the program contains any jobs.
        """
        request = quantum.DeleteQuantumProgramRequest(
            name=_program_name_from_ids(project_id, program_id), delete_jobs=delete_jobs
        )
        await self.grpc_client.delete_quantum_program(request)

    def delete_program(self, project_id: str, program_id: str, delete_jobs: bool = False) -> None:
        """Deletes a previously created quantum program. Prefer delete_program_async()."""
        request = quantum.DeleteQuantumProgramRequest(
            name=_program_name_from_ids(project_id, program_id), delete_jobs=delete_jobs
        )
        self._run_isolated_sync('delete_quantum_program', request)

    # --- Create Job ---

    def _build_quantum_job(
        self,
        *,
        project_id: str,
        program_id: str,
        job_id: str,
        processor_id: str = "",
        run_context: any_pb2.Any,
        priority: int | None = None,
        description: str | None = None,
        labels: dict[str, str] | None = None,
        run_name: str = "",
        snapshot_id: str = "",
        device_config_name: str = "",
    ) -> quantum.QuantumJob:
        if snapshot_id:
            selector = quantum.DeviceConfigSelector(
                snapshot_id=snapshot_id or None, config_alias=device_config_name
            )
        else:
            selector = quantum.DeviceConfigSelector(
                run_name=run_name or None, config_alias=device_config_name
            )
        job_name = _job_name_from_ids(project_id, program_id, job_id) if job_id else ''
        job = quantum.QuantumJob(
            name=job_name,
            scheduling_config=quantum.SchedulingConfig(
                processor_selector=quantum.SchedulingConfig.ProcessorSelector(
                    processor=_processor_name_from_ids(project_id, processor_id),
                    device_config_selector=selector,
                )
            ),
            run_context=run_context,
        )
        if priority:
            job.scheduling_config.priority = priority
        if description:
            job.description = description
        if labels:
            job.labels.update(labels)
        return job

    def _build_create_job_request(
        self,
        project_id,
        program_id,
        job_id,
        processor_id,
        run_context,
        priority,
        description,
        labels,
        run_name,
        snapshot_id,
        device_config_name,
    ) -> quantum.CreateQuantumJobRequest:
        # Check program to run and program parameters.
        if priority and not 0 <= priority < 1000:
            raise ValueError('priority must be between 0 and 1000')
        if not processor_id:
            raise ValueError('Must specify a processor id when creating a job.')
        if run_name and snapshot_id:
            raise ValueError('Cannot specify both `run_name` and `snapshot_id`')
        if (bool(run_name) or bool(snapshot_id)) ^ bool(device_config_name):
            raise ValueError(
                'Cannot specify only one of top level identifier (e.g `run_name`, `snapshot_id`)'
                ' and `device_config_name`'
            )

        job = self._build_quantum_job(
            project_id=project_id,
            program_id=program_id,
            job_id=job_id,
            processor_id=processor_id,
            run_context=run_context,
            priority=priority,
            description=description,
            labels=labels,
            run_name=run_name,
            snapshot_id=snapshot_id,
            device_config_name=device_config_name,
        )

        return quantum.CreateQuantumJobRequest(
            parent=_program_name_from_ids(project_id, program_id), quantum_job=job
        )

    async def create_job_async(
        self,
        project_id: str,
        program_id: str,
        job_id: str | None,
        processor_id: str,
        run_context: any_pb2.Any = any_pb2.Any(),
        priority: int | None = None,
        description: str | None = None,
        labels: dict[str, str] | None = None,
        *,
        run_name: str = "",
        snapshot_id: str = "",
        device_config_name: str = "",
    ) -> tuple[str, quantum.QuantumJob]:
        """Creates and runs a job on Quantum Engine.

        Either both `run_name` and `device_config_name` must be set, or neither
        of them must be set. If none of them are set, a default internal device
        configuration will be used.

        Args:
            project_id: A project_id of the parent Google Cloud Project.
            program_id: Unique ID of the program within the parent project.
            job_id: Unique ID of the job within the parent program.
            run_context: Properly serialized run context.
            priority: Optional priority to run at, 0-1000.
            description: Optional description to set on the job.
            labels: Optional set of labels to set on the job.
            processor_id: Processor id for running the program.
            run_name: A unique identifier representing an automation run for the
                specified processor. An Automation Run contains a collection of
                device configurations for a processor. If specified, `processor_id`
                is required to be set.
            snapshot_id: A unique identifier for an immutable snapshot reference.
                A snapshot contains a collection of device configurations for the
                processor.
            device_config_name: An identifier used to select the processor configuration
                utilized to run the job. A configuration identifies the set of
                available qubits, couplers, and supported gates in the processor.
                If specified, `processor_id` is required to be set.
        Returns:
            Tuple of created job id and job.

        Raises:
            ValueError: If the priority is not between 0 and 1000.
            ValueError: If  only one of `run_name` and `device_config_name` are specified.
            ValueError: If either `run_name` and `device_config_name` are set but
                `processor_id` is empty.
            ValueError: If both `run_name` and `snapshot_id` are specified.
        """
        request = self._build_create_job_request(
            project_id,
            program_id,
            job_id,
            processor_id,
            run_context,
            priority,
            description,
            labels,
            run_name,
            snapshot_id,
            device_config_name,
        )
        job = await self.grpc_client.create_quantum_job(request)
        return _ids_from_job_name(job.name)[2], job

    def create_job(
        self,
        project_id: str,
        program_id: str,
        job_id: str | None,
        processor_id: str,
        run_context: any_pb2.Any = any_pb2.Any(),
        priority: int | None = None,
        description: str | None = None,
        labels: dict[str, str] | None = None,
        *,
        run_name: str = "",
        snapshot_id: str = "",
        device_config_name: str = "",
    ) -> tuple[str, quantum.QuantumJob]:
        """Creates and runs a job on Quantum Engine. Prefer create_job_async()."""
        request = self._build_create_job_request(
            project_id,
            program_id,
            job_id,
            processor_id,
            run_context,
            priority,
            description,
            labels,
            run_name,
            snapshot_id,
            device_config_name,
        )
        job = self._run_isolated_sync('create_quantum_job', request)
        return _ids_from_job_name(job.name)[2], job

    # --- List Jobs ---

    def _build_list_jobs_request(
        self,
        project_id,
        program_id,
        created_before,
        created_after,
        has_labels,
        execution_states,
        executed_processor_ids,
        scheduled_processor_ids,
    ) -> quantum.ListQuantumJobsRequest:
        filters = []

        if created_after is not None:
            val = _date_or_time_to_filter_expr('created_after', created_after)
            filters.append(f"create_time >= {val}")
        if created_before is not None:
            val = _date_or_time_to_filter_expr('created_before', created_before)
            filters.append(f"create_time <= {val}")
        if has_labels is not None:
            for k, v in has_labels.items():
                filters.append(f"labels.{k}:{v}")
        if execution_states is not None:
            state_filter = []
            for execution_state in execution_states:
                state_filter.append(f"execution_status.state = {execution_state.name}")
            filters.append(f"({' OR '.join(state_filter)})")
        if executed_processor_ids is not None:
            ids_filter = []
            for processor_id in executed_processor_ids:
                ids_filter.append(f"executed_processor_id = {processor_id}")
            filters.append(f"({' OR '.join(ids_filter)})")
        if scheduled_processor_ids is not None:
            ids_filter = []
            for processor_id in scheduled_processor_ids:
                ids_filter.append(f"scheduled_processor_ids: {processor_id}")
            filters.append(f"({' OR '.join(ids_filter)})")

        if program_id is None:
            program_id = "-"
        parent = _program_name_from_ids(project_id, program_id)
        return quantum.ListQuantumJobsRequest(parent=parent, filter=" AND ".join(filters))

    async def list_jobs_async(
        self,
        project_id: str,
        program_id: str | None = None,
        created_before: datetime.datetime | datetime.date | None = None,
        created_after: datetime.datetime | datetime.date | None = None,
        has_labels: dict[str, str] | None = None,
        execution_states: set[quantum.ExecutionStatus.State] | None = None,
        executed_processor_ids: list[str] | None = None,
        scheduled_processor_ids: list[str] | None = None,
    ):
        """Returns the list of jobs for a given program.

        Args:
            project_id: A project_id of the parent Google Cloud Project.
            program_id: Optional, a unique ID of the program within the parent
                project. If None, jobs will be listed across all programs within
                the project.
            created_after: retrieve jobs that were created after this date
                or time.
            created_before: retrieve jobs that were created after this date
                or time.
            has_labels: retrieve jobs that have labels on them specified by
                this dict. If the value is set to `*`, filters having the label
                regardless of the label value will be filtered. For example, to
                query programs that have the shape label and have the color
                label with value red can be queried using

                {'color': 'red', 'shape':'*'}

            execution_states: retrieve jobs that have an execution state that
                is contained in `execution_states`. See
                `quantum.ExecutionStatus.State` enum for accepted values.

            executed_processor_ids: filters jobs by processor ID used for
                execution. Matches any of provided IDs.
            scheduled_processor_ids: filters jobs by any of provided
                scheduled processor IDs.
        """
        request = self._build_list_jobs_request(
            project_id,
            program_id,
            created_before,
            created_after,
            has_labels,
            execution_states,
            executed_processor_ids,
            scheduled_processor_ids,
        )
        return await self.grpc_client.list_quantum_jobs(request)

    def list_jobs(
        self,
        project_id: str,
        program_id: str | None = None,
        created_before: datetime.datetime | datetime.date | None = None,
        created_after: datetime.datetime | datetime.date | None = None,
        has_labels: dict[str, str] | None = None,
        execution_states: set[quantum.ExecutionStatus.State] | None = None,
        executed_processor_ids: list[str] | None = None,
        scheduled_processor_ids: list[str] | None = None,
    ):
        """Returns the list of jobs for a given program. Prefer list_jobs_async()."""
        request = self._build_list_jobs_request(
            project_id,
            program_id,
            created_before,
            created_after,
            has_labels,
            execution_states,
            executed_processor_ids,
            scheduled_processor_ids,
        )
        return self._run_isolated_sync('list_quantum_jobs', request, is_list_operation=True)

    # --- Get Job ---

    async def get_job_async(
        self, project_id: str, program_id: str, job_id: str, return_run_context: bool
    ) -> quantum.QuantumJob:
        """Returns a previously created job.

        Args:
            project_id: A project_id of the parent Google Cloud Project.
            program_id: Unique ID of the program within the parent project.
            job_id: Unique ID of the job within the parent program.
            return_run_context: If true then the run context will be loaded
                from the job's run_context_location and set on the returned
                QuantumJob.
        """
        request = quantum.GetQuantumJobRequest(
            name=_job_name_from_ids(project_id, program_id, job_id),
            return_run_context=return_run_context,
        )
        return await self.grpc_client.get_quantum_job(request)

    def get_job(
        self, project_id: str, program_id: str, job_id: str, return_run_context: bool
    ) -> quantum.QuantumJob:
        """Returns a previously created job. Prefer get_job_async()."""
        request = quantum.GetQuantumJobRequest(
            name=_job_name_from_ids(project_id, program_id, job_id),
            return_run_context=return_run_context,
        )
        return self._run_isolated_sync('get_quantum_job', request)

    # --- Set Job Description ---

    async def set_job_description_async(
        self, project_id: str, program_id: str, job_id: str, description: str
    ) -> quantum.QuantumJob:
        """Sets the description for a previously created quantum job.

        Args:
            project_id: A project_id of the parent Google Cloud Project.
            program_id: Unique ID of the program within the parent project.
            job_id: Unique ID of the job within the parent program.
            description: The new job description.

        Returns:
            The updated quantum job.
        """
        job_resource_name = _job_name_from_ids(project_id, program_id, job_id)
        request = quantum.UpdateQuantumJobRequest(
            name=job_resource_name,
            quantum_job=quantum.QuantumJob(name=job_resource_name, description=description),
            update_mask=field_mask_pb2.FieldMask(paths=['description']),
        )
        return await self.grpc_client.update_quantum_job(request)

    def set_job_description(
        self, project_id: str, program_id: str, job_id: str, description: str
    ) -> quantum.QuantumJob:
        """Sets the description for a previously created quantum job.
        Prefer set_job_description_async()."""
        job_resource_name = _job_name_from_ids(project_id, program_id, job_id)
        request = quantum.UpdateQuantumJobRequest(
            name=job_resource_name,
            quantum_job=quantum.QuantumJob(name=job_resource_name, description=description),
            update_mask=field_mask_pb2.FieldMask(paths=['description']),
        )
        return self._run_isolated_sync('update_quantum_job', request)

    # --- Set Job Labels ---

    def _build_set_job_labels_request(
        self, project_id, program_id, job_id, labels, fingerprint
    ) -> quantum.UpdateQuantumJobRequest:
        job_resource_name = _job_name_from_ids(project_id, program_id, job_id)
        return quantum.UpdateQuantumJobRequest(
            name=job_resource_name,
            quantum_job=quantum.QuantumJob(
                name=job_resource_name, labels=labels, label_fingerprint=fingerprint
            ),
            update_mask=field_mask_pb2.FieldMask(paths=['labels']),
        )

    async def _set_job_labels_async(
        self,
        project_id: str,
        program_id: str,
        job_id: str,
        labels: dict[str, str],
        fingerprint: str,
    ) -> quantum.QuantumJob:
        request = self._build_set_job_labels_request(
            project_id, program_id, job_id, labels, fingerprint
        )
        return await self.grpc_client.update_quantum_job(request)

    async def set_job_labels_async(
        self, project_id: str, program_id: str, job_id: str, labels: dict[str, str]
    ) -> quantum.QuantumJob:
        """Sets (overwriting) the labels for a previously created quantum job.

        Args:
            project_id: A project_id of the parent Google Cloud Project.
            program_id: Unique ID of the program within the parent project.
            job_id: Unique ID of the job within the parent program.
            labels: The entire set of new job labels.

        Returns:
            The updated quantum job.
        """
        job = await self.get_job_async(project_id, program_id, job_id, False)
        return await self._set_job_labels_async(
            project_id, program_id, job_id, labels, job.label_fingerprint
        )

    def _set_job_labels(
        self,
        project_id: str,
        program_id: str,
        job_id: str,
        labels: dict[str, str],
        fingerprint: str,
    ) -> quantum.QuantumJob:
        request = self._build_set_job_labels_request(
            project_id, program_id, job_id, labels, fingerprint
        )
        return self._run_isolated_sync('update_quantum_job', request)

    def set_job_labels(
        self, project_id: str, program_id: str, job_id: str, labels: dict[str, str]
    ) -> quantum.QuantumJob:
        """Sets (overwriting) the labels for a previously created quantum job.
        Prefer set_job_labels_async()."""
        job = self.get_job(project_id, program_id, job_id, False)
        return self._set_job_labels(project_id, program_id, job_id, labels, job.label_fingerprint)

    # --- Add Job Labels ---

    async def add_job_labels_async(
        self, project_id: str, program_id: str, job_id: str, labels: dict[str, str]
    ) -> quantum.QuantumJob:
        """Adds new labels to a previously created quantum job.

        Args:
            project_id: A project_id of the parent Google Cloud Project.
            program_id: Unique ID of the program within the parent project.
            job_id: Unique ID of the job within the parent program.
            labels: New labels to add to the existing job labels.

        Returns:
            The updated quantum job.
        """
        job = await self.get_job_async(project_id, program_id, job_id, False)
        old_labels = job.labels
        new_labels = dict(old_labels)
        new_labels.update(labels)
        if new_labels != old_labels:
            fingerprint = job.label_fingerprint
            return await self._set_job_labels_async(
                project_id, program_id, job_id, new_labels, fingerprint
            )
        return job

    def add_job_labels(
        self, project_id: str, program_id: str, job_id: str, labels: dict[str, str]
    ) -> quantum.QuantumJob:
        """Adds new labels to a previously created quantum job. Prefer add_job_labels_async()."""
        job = self.get_job(project_id, program_id, job_id, False)
        old_labels = job.labels
        new_labels = dict(old_labels)
        new_labels.update(labels)
        if new_labels != old_labels:
            fingerprint = job.label_fingerprint
            return self._set_job_labels(project_id, program_id, job_id, new_labels, fingerprint)
        return job

    # --- Remove Job Labels ---

    async def remove_job_labels_async(
        self, project_id: str, program_id: str, job_id: str, label_keys: list[str]
    ) -> quantum.QuantumJob:
        """Removes labels with given keys from the labels of a previously
        created quantum job.

        Args:
            project_id: A project_id of the parent Google Cloud Project.
            program_id: Unique ID of the program within the parent project.
            job_id: Unique ID of the job within the parent program.
            label_keys: Label keys to remove from the existing job labels.

        Returns:
            The updated quantum job.
        """
        job = await self.get_job_async(project_id, program_id, job_id, False)
        old_labels = job.labels
        new_labels = dict(old_labels)
        for key in label_keys:
            new_labels.pop(key, None)
        if new_labels != old_labels:
            fingerprint = job.label_fingerprint
            return await self._set_job_labels_async(
                project_id, program_id, job_id, new_labels, fingerprint
            )
        return job

    def remove_job_labels(
        self, project_id: str, program_id: str, job_id: str, label_keys: list[str]
    ) -> quantum.QuantumJob:
        """Removes labels with given keys from the labels of a previously
        created quantum job. Prefer remove_job_labels_async()."""
        job = self.get_job(project_id, program_id, job_id, False)
        old_labels = job.labels
        new_labels = dict(old_labels)
        for key in label_keys:
            new_labels.pop(key, None)
        if new_labels != old_labels:
            fingerprint = job.label_fingerprint
            return self._set_job_labels(project_id, program_id, job_id, new_labels, fingerprint)
        return job

    # --- Delete Job ---

    async def delete_job_async(self, project_id: str, program_id: str, job_id: str) -> None:
        """Deletes a previously created quantum job.

        Args:
            project_id: A project_id of the parent Google Cloud Project.
            program_id: Unique ID of the program within the parent project.
            job_id: Unique ID of the job within the parent program.
        """
        request = quantum.DeleteQuantumJobRequest(
            name=_job_name_from_ids(project_id, program_id, job_id)
        )
        await self.grpc_client.delete_quantum_job(request)

    def delete_job(self, project_id: str, program_id: str, job_id: str) -> None:
        """Deletes a previously created quantum job. Prefer delete_job_async()."""
        request = quantum.DeleteQuantumJobRequest(
            name=_job_name_from_ids(project_id, program_id, job_id)
        )
        self._run_isolated_sync('delete_quantum_job', request)

    # --- Cancel Job ---

    async def cancel_job_async(self, project_id: str, program_id: str, job_id: str) -> None:
        """Cancels the given job.

        Args:
            project_id: A project_id of the parent Google Cloud Project.
            program_id: Unique ID of the program within the parent project.
            job_id: Unique ID of the job within the parent program.
        """
        request = quantum.CancelQuantumJobRequest(
            name=_job_name_from_ids(project_id, program_id, job_id)
        )
        await self.grpc_client.cancel_quantum_job(request)

    def cancel_job(self, project_id: str, program_id: str, job_id: str) -> None:
        """Cancels the given job. Prefer cancel_job_async()."""
        request = quantum.CancelQuantumJobRequest(
            name=_job_name_from_ids(project_id, program_id, job_id)
        )
        self._run_isolated_sync('cancel_quantum_job', request)

    # --- Get Job Results ---

    async def get_job_results_async(
        self, project_id: str, program_id: str, job_id: str
    ) -> quantum.QuantumResult:
        """Returns the results of a completed job.

        Args:
            project_id: A project_id of the parent Google Cloud Project.
            program_id: Unique ID of the program within the parent project.
            job_id: Unique ID of the job within the parent program.

        Returns:
            The quantum result.
        """
        request = quantum.GetQuantumResultRequest(
            parent=_job_name_from_ids(project_id, program_id, job_id)
        )
        return await self.grpc_client.get_quantum_result(request)

    def get_job_results(
        self, project_id: str, program_id: str, job_id: str
    ) -> quantum.QuantumResult:
        """Returns the results of a completed job. Prefer get_job_results_async()."""
        request = quantum.GetQuantumResultRequest(
            parent=_job_name_from_ids(project_id, program_id, job_id)
        )
        return self._run_isolated_sync('get_quantum_result', request)

    # --- Run Job Over Stream ---

    def run_job_over_stream(
        self,
        *,
        project_id: str,
        program_id: str,
        code: any_pb2.Any,
        run_context: any_pb2.Any,
        program_description: str | None = None,
        program_labels: dict[str, str] | None = None,
        job_id: str,
        priority: int | None = None,
        job_description: str | None = None,
        job_labels: dict[str, str] | None = None,
        processor_id: str = "",
        run_name: str = "",
        snapshot_id: str = "",
        device_config_name: str = "",
    ) -> concurrent.futures.Future[quantum.QuantumResult | quantum.QuantumJob]:
        """Runs a job with the given program and job information over a stream.

        Sends the request over the Quantum Engine QuantumRunStream bidirectional stream, and returns
        a future for the stream response. The future will be completed with a `QuantumResult` if
        the job is successful; otherwise, it will be completed with a QuantumJob.

        Args:
            project_id: A project_id of the parent Google Cloud Project.
            program_id: Unique ID of the program within the parent project.
            code: Properly serialized program code.
            run_context: Properly serialized run context.
            program_description: An optional description to set on the program.
            program_labels: Optional set of labels to set on the program.
            job_id: Unique ID of the job within the parent program.
            priority: Optional priority to run at, 0-1000.
            job_description: Optional description to set on the job.
            job_labels: Optional set of labels to set on the job.
            processor_id: Processor id for running the program.
            run_name: A unique identifier representing an automation run for the
                specified processor. An Automation Run contains a collection of
                device configurations for a processor. If specified, `processor_id`
                is required to be set.
            snapshot_id: A unique identifier for an immutable snapshot reference.
                A snapshot contains a collection of device configurations for the
                processor.
            device_config_name: An identifier used to select the processor configuration
                utilized to run the job. A configuration identifies the set of
                available qubits, couplers, and supported gates in the processor.
                If specified, `processor_id` is required to be set.

        Returns:
            A future for the job result, or the job if the job has failed.

        Raises:
            ValueError: If the priority is not between 0 and 1000.
            ValueError: If `processor_id` is not set.
            ValueError: If only one of `run_name` and `device_config_name` are specified.
            ValueError: If both `run_name` and `snapshot_id` are specified.
        """
        # Check program to run and program parameters.
        if priority and not 0 <= priority < 1000:
            raise ValueError('priority must be between 0 and 1000')
        if not processor_id:
            raise ValueError('Must specify a processor id when creating a job.')
        if run_name and snapshot_id:
            raise ValueError('Cannot specify both `run_name` and `snapshot_id`')
        if (bool(run_name) or bool(snapshot_id)) ^ bool(device_config_name):
            raise ValueError(
                'Cannot specify only one of top level identifier and `device_config_name`'
            )

        project_name = _project_name(project_id)

        program_name = _program_name_from_ids(project_id, program_id)
        program = quantum.QuantumProgram(name=program_name, code=code)
        if program_description:
            program.description = program_description
        if program_labels:
            program.labels.update(program_labels)

        job = self._build_quantum_job(
            project_id=project_id,
            program_id=program_id,
            job_id=job_id,
            processor_id=processor_id,
            run_context=run_context,
            priority=priority,
            description=job_description,
            labels=job_labels,
            run_name=run_name,
            snapshot_id=snapshot_id,
            device_config_name=device_config_name,
        )
        return self._stream_manager.submit(project_name, program, job)

    # --- List Processors ---

    async def list_processors_async(self, project_id: str) -> list[quantum.QuantumProcessor]:
        """Returns a list of Processors that the user has visibility to in the
        current Engine project. The names of these processors are used to
        identify devices when scheduling jobs and gathering calibration metrics.

        Args:
            project_id: A project_id of the parent Google Cloud Project.

        Returns:
            A list of metadata of each processor.
        """
        request = quantum.ListQuantumProcessorsRequest(parent=_project_name(project_id), filter='')
        pager = await self.grpc_client.list_quantum_processors(request)
        return [processor async for processor in pager]

    def list_processors(self, project_id: str) -> list[quantum.QuantumProcessor]:
        """Returns a list of Processors that the user has visibility to in the
        current Engine project. Prefer list_processors_async()."""
        request = quantum.ListQuantumProcessorsRequest(parent=_project_name(project_id), filter='')
        return self._run_isolated_sync('list_quantum_processors', request, is_list_operation=True)

    # --- Get Processor ---

    async def get_processor_async(
        self, project_id: str, processor_id: str
    ) -> quantum.QuantumProcessor:
        """Returns a quantum processor.

        Args:
            project_id: A project_id of the parent Google Cloud Project.
            processor_id: The processor unique identifier.

        Returns:
            The quantum processor.
        """
        request = quantum.GetQuantumProcessorRequest(
            name=_processor_name_from_ids(project_id, processor_id)
        )
        return await self.grpc_client.get_quantum_processor(request)

    def get_processor(self, project_id: str, processor_id: str) -> quantum.QuantumProcessor:
        """Returns a quantum processor. Prefer get_processor_async()."""
        request = quantum.GetQuantumProcessorRequest(
            name=_processor_name_from_ids(project_id, processor_id)
        )
        return self._run_isolated_sync('get_quantum_processor', request)

    # --- List Calibrations ---

    async def list_calibrations_async(
        self, project_id: str, processor_id: str, filter_str: str = ''
    ) -> list[quantum.QuantumCalibration]:
        """Returns a list of quantum calibrations.

        Args:
            project_id: A project_id of the parent Google Cloud Project.
            processor_id: The processor unique identifier.
            filter_str: Filter string current only supports 'timestamp' with values
            of epoch time in seconds or short string 'yyyy-MM-dd'. For example:
                'timestamp > 1577960125 AND timestamp <= 1578241810'
                'timestamp > 2020-01-02 AND timestamp <= 2020-01-05'

        Returns:
            A list of calibrations.
        """
        request = quantum.ListQuantumCalibrationsRequest(
            parent=_processor_name_from_ids(project_id, processor_id), filter=filter_str
        )
        pager = await self.grpc_client.list_quantum_calibrations(request)
        return [calibration async for calibration in pager]

    def list_calibrations(
        self, project_id: str, processor_id: str, filter_str: str = ''
    ) -> list[quantum.QuantumCalibration]:
        """Returns a list of quantum calibrations. Prefer list_calibrations_async()."""
        request = quantum.ListQuantumCalibrationsRequest(
            parent=_processor_name_from_ids(project_id, processor_id), filter=filter_str
        )
        return self._run_isolated_sync('list_quantum_calibrations', request, is_list_operation=True)

    # --- Get Calibration ---

    async def get_calibration_async(
        self, project_id: str, processor_id: str, calibration_timestamp_seconds: int
    ) -> quantum.QuantumCalibration:
        """Returns a quantum calibration.

        Args:
            project_id: A project_id of the parent Google Cloud Project.
            processor_id: The processor unique identifier.
            calibration_timestamp_seconds: The timestamp of the calibration in
                seconds.

        Returns:
            The quantum calibration.
        """
        request = quantum.GetQuantumCalibrationRequest(
            name=_calibration_name_from_ids(project_id, processor_id, calibration_timestamp_seconds)
        )
        return await self.grpc_client.get_quantum_calibration(request)

    def get_calibration(
        self, project_id: str, processor_id: str, calibration_timestamp_seconds: int
    ) -> quantum.QuantumCalibration:
        """Returns a quantum calibration. Prefer get_calibration_async()."""
        request = quantum.GetQuantumCalibrationRequest(
            name=_calibration_name_from_ids(project_id, processor_id, calibration_timestamp_seconds)
        )
        return self._run_isolated_sync('get_quantum_calibration', request)

    # --- Get Current Calibration ---

    async def get_current_calibration_async(
        self, project_id: str, processor_id: str
    ) -> quantum.QuantumCalibration | None:
        """Returns the current quantum calibration for a processor if it has one.

        Args:
            project_id: A project_id of the parent Google Cloud Project.
            processor_id: The processor unique identifier.

        Returns:
            The quantum calibration or None if there is no current calibration.

        Raises:
            EngineException: If the request for calibration fails.
        """
        try:
            request = quantum.GetQuantumCalibrationRequest(
                name=_processor_name_from_ids(project_id, processor_id) + '/calibrations/current'
            )
            return await self.grpc_client.get_quantum_calibration(request)
        except EngineException as err:
            if isinstance(err.__cause__, NotFound):
                return None
            raise

    def get_current_calibration(
        self, project_id: str, processor_id: str
    ) -> quantum.QuantumCalibration | None:
        """Returns the current quantum calibration for a processor if it has one.
        Prefer get_current_calibration_async()."""
        try:
            request = quantum.GetQuantumCalibrationRequest(
                name=_processor_name_from_ids(project_id, processor_id) + '/calibrations/current'
            )
            return self._run_isolated_sync('get_quantum_calibration', request)
        except EngineException as err:
            if isinstance(err.__cause__, NotFound):
                return None
            raise

    # --- Create Reservation ---

    def _build_create_reservation_request(
        self,
        project_id: str,
        processor_id: str,
        start: datetime.datetime,
        end: datetime.datetime,
        allowlisted_users: list[str] | None = None,
    ) -> quantum.CreateQuantumReservationRequest:
        parent = _processor_name_from_ids(project_id, processor_id)
        reservation = quantum.QuantumReservation(
            name='',
            start_time=Timestamp(seconds=int(start.timestamp())),
            end_time=Timestamp(seconds=int(end.timestamp())),
        )
        if allowlisted_users:
            reservation.allowlisted_users.extend(allowlisted_users)
        return quantum.CreateQuantumReservationRequest(
            parent=parent, quantum_reservation=reservation
        )

    @_compat.deprecated_parameter(
        deadline='v1.7',
        fix='Change whitelisted_users to allowlisted_users.',
        parameter_desc='whitelisted_users',
        match=lambda args, kwargs: 'whitelisted_users' in kwargs,
        rewrite=_fix_deprecated_allowlisted_users_args,
    )
    async def create_reservation_async(
        self,
        project_id: str,
        processor_id: str,
        start: datetime.datetime,
        end: datetime.datetime,
        allowlisted_users: list[str] | None = None,
    ):
        """Creates a quantum reservation and returns the created object.

        Args:
            project_id: A project_id of the parent Google Cloud Project.
            processor_id: The processor unique identifier.
            reservation_id: Unique ID of the reservation in the parent project,
                or None if the engine should generate an id
            start: the starting time of the reservation as a datetime object
            end: the ending time of the reservation as a datetime object
            allowlisted_users: a list of emails that can use the reservation.
        """
        request = self._build_create_reservation_request(
            project_id, processor_id, start, end, allowlisted_users
        )
        return await self.grpc_client.create_quantum_reservation(request)

    @_compat.deprecated_parameter(
        deadline='v1.7',
        fix='Change whitelisted_users to allowlisted_users.',
        parameter_desc='whitelisted_users',
        match=lambda args, kwargs: 'whitelisted_users' in kwargs,
        rewrite=_fix_deprecated_allowlisted_users_args,
    )
    def create_reservation(
        self,
        project_id: str,
        processor_id: str,
        start: datetime.datetime,
        end: datetime.datetime,
        allowlisted_users: list[str] | None = None,
    ):
        """Creates a quantum reservation and returns the created object.
        Prefer create_reservation_async()."""
        request = self._build_create_reservation_request(
            project_id, processor_id, start, end, allowlisted_users
        )
        return self._run_isolated_sync('create_quantum_reservation', request)

    # --- Cancel Reservation ---

    async def cancel_reservation_async(
        self, project_id: str, processor_id: str, reservation_id: str
    ):
        """Cancels a quantum reservation.

        This action is only valid if the associated [QuantumProcessor]
        schedule not been frozen. Otherwise, delete_reservation should
        be used.

        The reservation will be truncated to end at the time when the request is
        serviced and any remaining time will be made available as an open swim
        period. This action will only succeed if the reservation has not yet
        ended and is within the processor's freeze window. If the reservation
        has already ended or is beyond the processor's freeze window, then the
        call will return an error.

        Args:
            project_id: A project_id of the parent Google Cloud Project.
            processor_id: The processor unique identifier.
            reservation_id: Unique ID of the reservation in the parent project,
        """
        name = _reservation_name_from_ids(project_id, processor_id, reservation_id)
        request = quantum.CancelQuantumReservationRequest(name=name)
        return await self.grpc_client.cancel_quantum_reservation(request)

    def cancel_reservation(self, project_id: str, processor_id: str, reservation_id: str):
        """Cancels a quantum reservation.

        This action is only valid if the associated [QuantumProcessor]
        schedule not been frozen. Otherwise, delete_reservation should
        be used.

        The reservation will be truncated to end at the time when the request is
        serviced and any remaining time will be made available as an open swim
        period. This action will only succeed if the reservation has not yet
        ended and is within the processor's freeze window. If the reservation
        has already ended or is beyond the processor's freeze window, then the
        call will return an error.

        Args:
            project_id: A project_id of the parent Google Cloud Project.
            processor_id: The processor unique identifier.
            reservation_id: Unique ID of the reservation in the parent project,
        """
        name = _reservation_name_from_ids(project_id, processor_id, reservation_id)
        request = quantum.CancelQuantumReservationRequest(name=name)
        return self._run_isolated_sync('cancel_quantum_reservation', request)

    # --- Delete Reservation ---

    async def delete_reservation_async(
        self, project_id: str, processor_id: str, reservation_id: str
    ):
        """Deletes a quantum reservation.

        This action is only valid if the associated [QuantumProcessor]
        schedule has not been frozen.  Otherwise, cancel_reservation
        should be used.

        If the reservation has already ended or is within the processor's
        freeze window, then the call will return a `FAILED_PRECONDITION` error.

        Args:
            project_id: A project_id of the parent Google Cloud Project.
            processor_id: The processor unique identifier.
            reservation_id: Unique ID of the reservation in the parent project,
        """
        name = _reservation_name_from_ids(project_id, processor_id, reservation_id)
        request = quantum.DeleteQuantumReservationRequest(name=name)
        return await self.grpc_client.delete_quantum_reservation(request)

    def delete_reservation(self, project_id: str, processor_id: str, reservation_id: str):
        """Deletes a quantum reservation. Prefer delete_reservation_async()."""
        name = _reservation_name_from_ids(project_id, processor_id, reservation_id)
        request = quantum.DeleteQuantumReservationRequest(name=name)
        return self._run_isolated_sync('delete_quantum_reservation', request)

    # --- Get Reservation ---

    async def get_reservation_async(
        self, project_id: str, processor_id: str, reservation_id: str
    ) -> quantum.QuantumReservation | None:
        """Gets a quantum reservation from the engine.

        Args:
            project_id: A project_id of the parent Google Cloud Project.
            processor_id: The processor unique identifier.
            reservation_id: Unique ID of the reservation in the parent project.

        Raises:
            EngineException: If the request to get the reservation failed.
        """
        try:
            name = _reservation_name_from_ids(project_id, processor_id, reservation_id)
            request = quantum.GetQuantumReservationRequest(name=name)
            return await self.grpc_client.get_quantum_reservation(request)
        except EngineException as err:
            if isinstance(err.__cause__, NotFound):
                return None
            raise

    def get_reservation(
        self, project_id: str, processor_id: str, reservation_id: str
    ) -> quantum.QuantumReservation | None:
        """Gets a quantum reservation from the engine. Prefer get_reservation_async()."""
        try:
            name = _reservation_name_from_ids(project_id, processor_id, reservation_id)
            request = quantum.GetQuantumReservationRequest(name=name)
            return self._run_isolated_sync('get_quantum_reservation', request)
        except EngineException as err:
            if isinstance(err.__cause__, NotFound):
                return None
            raise

    # --- List Reservations ---

    async def list_reservations_async(
        self, project_id: str, processor_id: str, filter_str: str = ''
    ) -> list[quantum.QuantumReservation]:
        """Returns a list of quantum reservations.

        Only reservations owned by this project will be returned.

        Args:
            project_id: A project_id of the parent Google Cloud Project.
            processor_id: The processor unique identifier.
            filter_str: A string for filtering quantum reservations.
                The fields eligible for filtering are start_time and end_time
                Examples:
                    `start_time >= 1584385200`: Reservation began on or after
                        the epoch time Mar 16th, 7pm GMT.
                    `end_time >= 1483370475`: Reservation ends on
                        or after Jan 2nd 2017 15:21:15

        Returns:
            A list of QuantumReservation objects.
        """
        request = quantum.ListQuantumReservationsRequest(
            parent=_processor_name_from_ids(project_id, processor_id), filter=filter_str
        )
        pager = await self.grpc_client.list_quantum_reservations(request)
        return [reservation async for reservation in pager]

    def list_reservations(
        self, project_id: str, processor_id: str, filter_str: str = ''
    ) -> list[quantum.QuantumReservation]:
        """Returns a list of quantum reservations. Prefer list_reservations_async()."""
        request = quantum.ListQuantumReservationsRequest(
            parent=_processor_name_from_ids(project_id, processor_id), filter=filter_str
        )
        return self._run_isolated_sync('list_quantum_reservations', request, is_list_operation=True)

    # --- Update Reservation ---

    def _build_update_reservation_request(
        self, project_id, processor_id, reservation_id, start, end, allowlisted_users
    ):
        name = (
            _reservation_name_from_ids(project_id, processor_id, reservation_id)
            if reservation_id
            else ''
        )

        reservation = quantum.QuantumReservation(name=name)
        paths = []
        if start:
            reservation.start_time = start
            paths.append('start_time')
        if end:
            reservation.end_time = end
            paths.append('end_time')
        if allowlisted_users is not None:
            reservation.allowlisted_users.extend(allowlisted_users)
            paths.append('allowlisted_users')

        return quantum.UpdateQuantumReservationRequest(
            name=name,
            quantum_reservation=reservation,
            update_mask=field_mask_pb2.FieldMask(paths=paths),
        )

    @_compat.deprecated_parameter(
        deadline='v1.7',
        fix='Change whitelisted_users to allowlisted_users.',
        parameter_desc='whitelisted_users',
        match=lambda args, kwargs: 'whitelisted_users' in kwargs,
        rewrite=_fix_deprecated_allowlisted_users_args,
    )
    async def update_reservation_async(
        self,
        project_id: str,
        processor_id: str,
        reservation_id: str,
        start: datetime.datetime | None = None,
        end: datetime.datetime | None = None,
        allowlisted_users: list[str] | None = None,
    ):
        """Updates a quantum reservation.

        This will update a quantum reservation's starting time, ending time,
        and list of allowlisted users.  If any field is not filled, it will
        not be updated.

        Args:
            project_id: A project_id of the parent Google Cloud Project.
            processor_id: The processor unique identifier.
            reservation_id: Unique ID of the reservation in the parent project,
            start: the new starting time of the reservation as a datetime object
            end: the new ending time of the reservation as a datetime object
            allowlisted_users: a list of emails that can use the reservation.
                The empty list, [], will clear the allowlisted_users while None
                will leave the value unchanged.
        """
        request = self._build_update_reservation_request(
            project_id, processor_id, reservation_id, start, end, allowlisted_users
        )
        return await self.grpc_client.update_quantum_reservation(request)

    @_compat.deprecated_parameter(
        deadline='v1.7',
        fix='Change whitelisted_users to allowlisted_users.',
        parameter_desc='whitelisted_users',
        match=lambda args, kwargs: 'whitelisted_users' in kwargs,
        rewrite=_fix_deprecated_allowlisted_users_args,
    )
    def update_reservation(
        self,
        project_id: str,
        processor_id: str,
        reservation_id: str,
        start: datetime.datetime | None = None,
        end: datetime.datetime | None = None,
        allowlisted_users: list[str] | None = None,
    ):
        """Updates a quantum reservation. Prefer update_reservation_async()."""
        request = self._build_update_reservation_request(
            project_id, processor_id, reservation_id, start, end, allowlisted_users
        )
        return self._run_isolated_sync('update_quantum_reservation', request)

    # --- List Time Slots ---

    async def list_time_slots_async(
        self, project_id: str, processor_id: str, filter_str: str = ''
    ) -> list[quantum.QuantumTimeSlot]:
        """Returns a list of quantum time slots on a processor.

        Args:
            project_id: A project_id of the parent Google Cloud Project.
            processor_id: The processor unique identifier.
            filter_str:  A string expression for filtering the quantum
                time slots returned by the list command. The fields
                eligible for filtering are `start_time`, `end_time`.

        Returns:
            A list of QuantumTimeSlot objects.
        """
        request = quantum.ListQuantumTimeSlotsRequest(
            parent=_processor_name_from_ids(project_id, processor_id), filter=filter_str
        )
        pager = await self.grpc_client.list_quantum_time_slots(request)
        return [time_slot async for time_slot in pager]

    def list_time_slots(
        self, project_id: str, processor_id: str, filter_str: str = ''
    ) -> list[quantum.QuantumTimeSlot]:
        """Returns a list of quantum time slots on a processor. Prefer list_time_slots_async()."""
        request = quantum.ListQuantumTimeSlotsRequest(
            parent=_processor_name_from_ids(project_id, processor_id), filter=filter_str
        )
        return self._run_isolated_sync('list_quantum_time_slots', request, is_list_operation=True)

    # --- Get Quantum Processor Config ---

    def _build_get_quantum_processor_config_request(
        self, project_id, processor_id, config_name, device_config_revision
    ):
        config_revision = _quantum_processor_revision_path(
            project_id=project_id,
            processor_id=processor_id,
            device_config_revision=device_config_revision,
        )
        return quantum.GetQuantumProcessorConfigRequest(
            name=f'{config_revision}/configs/{config_name}'
        )

    async def get_quantum_processor_config_async(
        self,
        project_id: str,
        processor_id: str,
        config_name: str = 'default',
        device_config_revision: DeviceConfigRevision = Run(id='current'),
    ) -> quantum.QuantumProcessorConfig | None:
        """Returns the QuantumProcessorConfig for the given snapshot id.

        Args:
            project_id: A project_id of the parent Google Cloud Project.
            processor_id: The processor unique identifier.
            device_config_revision: Specifies either the snapshot_id or the run_name.
            config_name: The id of the quantum processor config.

        Returns:
            The quantum procesor config or None if it does not exist.

        Raises:
            EngineException: If the request to get the config fails.
        """
        try:
            request = self._build_get_quantum_processor_config_request(
                project_id, processor_id, config_name, device_config_revision
            )
            return await self.grpc_client.get_quantum_processor_config(request)
        except EngineException as err:
            if isinstance(err.__cause__, NotFound):
                return None
            raise

    def get_quantum_processor_config(
        self,
        project_id: str,
        processor_id: str,
        config_name: str = 'default',
        device_config_revision: DeviceConfigRevision = Run(id='current'),
    ) -> quantum.QuantumProcessorConfig | None:
        """Returns the QuantumProcessorConfig for the given snapshot id.
        Prefer get_quantum_processor_config_async()."""
        try:
            request = self._build_get_quantum_processor_config_request(
                project_id, processor_id, config_name, device_config_revision
            )
            return self._run_isolated_sync('get_quantum_processor_config', request)
        except EngineException as err:
            if isinstance(err.__cause__, NotFound):
                return None
            raise

    # --- List Quantum Processor Configs ---

    def _build_list_quantum_processor_configs_request(
        self, project_id, processor_id, device_config_revision
    ):
        parent_resource_name = _quantum_processor_revision_path(
            project_id=project_id,
            processor_id=processor_id,
            device_config_revision=device_config_revision,
        )
        return quantum.ListQuantumProcessorConfigsRequest(parent=parent_resource_name)

    async def list_quantum_processor_configs_async(
        self,
        project_id: str,
        processor_id: str,
        device_config_revision: DeviceConfigRevision = Run(id='current'),
    ) -> list[quantum.QuantumProcessorConfig]:
        """Returns the QuantumProcessorConfig for the given snapshot id.

        Args:
            project_id: A project_id of the parent Google Cloud Project.
            processor_id: The processor unique identifier.
            device_config_revision: Specifies either the snapshot_id or the run_name.

        Returns:
            List of quantum procesor configs.
        """
        request = self._build_list_quantum_processor_configs_request(
            project_id, processor_id, device_config_revision
        )
        pager = await self.grpc_client.list_quantum_processor_configs(request)
        return [config async for config in pager]

    def list_quantum_processor_configs(
        self,
        project_id: str,
        processor_id: str,
        device_config_revision: DeviceConfigRevision = Run(id='current'),
    ) -> list[quantum.QuantumProcessorConfig]:
        """Returns the QuantumProcessorConfig for the given snapshot id.
        Prefer list_quantum_processor_configs_async()."""
        request = self._build_list_quantum_processor_configs_request(
            project_id, processor_id, device_config_revision
        )
        return self._run_isolated_sync(
            'list_quantum_processor_configs', request, is_list_operation=True
        )


def _project_name(project_id: str) -> str:
    return f'projects/{project_id}'


def _program_name_from_ids(project_id: str, program_id: str) -> str:
    return f'projects/{project_id}/programs/{program_id}'


def _job_name_from_ids(project_id: str, program_id: str, job_id: str) -> str:
    return f'projects/{project_id}/programs/{program_id}/jobs/{job_id}'


def _processor_name_from_ids(project_id: str, processor_id: str) -> str:
    return f'projects/{project_id}/processors/{processor_id}'


def _calibration_name_from_ids(
    project_id: str, processor_id: str, calibration_time_seconds: int
) -> str:
    return (
        f'projects/{project_id}/processors/{processor_id}/calibrations/{calibration_time_seconds}'
    )


def _reservation_name_from_ids(project_id: str, processor_id: str, reservation_id: str) -> str:
    return f'projects/{project_id}/processors/{processor_id}/reservations/{reservation_id}'


def _ids_from_program_name(program_name: str) -> tuple[str, str]:
    parts = program_name.split('/')
    return parts[1], parts[3]


def _ids_from_job_name(job_name: str) -> tuple[str, str, str]:
    parts = job_name.split('/')
    return parts[1], parts[3], parts[5]


def _ids_from_processor_name(processor_name: str) -> tuple[str, str]:
    parts = processor_name.split('/')
    return parts[1], parts[3]


def _ids_from_calibration_name(calibration_name: str) -> tuple[str, str, int]:
    parts = calibration_name.split('/')
    return parts[1], parts[3], int(parts[5])


def _quantum_processor_revision_path(
    project_id: str, processor_id: str, device_config_revision: DeviceConfigRevision | None = None
) -> str:
    processor_resource_name = _processor_name_from_ids(project_id, processor_id)
    if isinstance(device_config_revision, Snapshot):
        return f'{processor_resource_name}/configSnapshots/{device_config_revision.id}'

    default_run_name = 'default'
    run_id = device_config_revision.id if device_config_revision else default_run_name
    return f'{processor_resource_name}/configAutomationRuns/{run_id}'


def _date_or_time_to_filter_expr(param_name: str, param: datetime.datetime | datetime.date):
    """Formats datetime or date to filter expressions.

    Args:
        param_name: The name of the filter parameter (for error messaging).
        param: The value of the parameter.

    Raises:
        ValueError: If the supplied param is not a datetime or date.
    """
    if isinstance(param, datetime.datetime):
        return f"{int(param.timestamp())}"
    elif isinstance(param, datetime.date):
        return f"{param.isoformat()}"

    raise ValueError(
        f"Unsupported date/time type for {param_name}: got {param} of "
        f"type {type(param)}. Supported types: datetime.datetime and"
        f"datetime.date"
    )
