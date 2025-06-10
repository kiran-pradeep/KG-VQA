import os
import json
from pathlib import Path
from multiprocessing import Manager, Pool
from typing import List, Dict, Any, Tuple
from langgraph.graph import StateGraph

from runner.logger import Logger
from runner.task import QDVerdict, Task

# from runner.database_manager import DatabaseManager
from runner.statistics_manager import StatisticsManager
from workflow.team_builder import build_team

# from database_utils.execution import ExecutionStatus
from workflow.system_state import SystemState
import fcntl

from tqdm.auto import tqdm


class RunManager:
    RESULT_ROOT_PATH = "results"

    def __init__(self, args: Any):
        self.args = args
        if args.result_directory:
            self.result_directory = args.result_directory
        else:
            self.result_directory = self.get_result_directory()
        self.statistics_manager = StatisticsManager(self.result_directory)
        self.tasks: List[Task] = []
        self.total_number_of_tasks = 0
        self.processed_tasks = 0

    def get_result_directory(self) -> str:
        """
        Creates and returns the result directory path based on the input arguments.

        Returns:
            str: The path to the result directory.
        """
        data_mode = self.args.data_mode
        setting_name = self.args.config["setting_name"]
        dataset_name = Path(self.args.data_path).stem
        run_folder_name = str(self.args.run_start_time)
        run_folder_path = (
            Path(self.RESULT_ROOT_PATH)
            / data_mode
            / setting_name
            / dataset_name
            / run_folder_name
        )

        run_folder_path.mkdir(parents=True, exist_ok=True)

        arg_file_path = run_folder_path / "-args.json"
        if not arg_file_path.exists():
            with arg_file_path.open("w") as file:
                json.dump(vars(self.args), file, indent=4)

        final_prediction_file = run_folder_path / "-predictions.json"
        if not final_prediction_file.exists():
            with final_prediction_file.open("w") as file:
                json.dump({}, file, indent=4)

        log_folder_path = run_folder_path / "logs"
        log_folder_path.mkdir(exist_ok=True)

        return str(run_folder_path)

    def update_final_predictions(
        self, question_id: int, final_response: str = None, db_id: int = None
    ):
        results = {}
        if final_response:
            temp_results = {str(question_id): final_response}
        else:
            temp_results = {str(question_id): None}
        file_path = os.path.join(self.result_directory, "-predictions.json")
        with open(file_path, "r+") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                results = json.load(f)
                results.update(temp_results)
                f.seek(0)
                json.dump(results, f, indent=4)
                f.truncate()
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)

    def initialize_tasks(self, dataset: List[Dict[str, Any]]):
        """
        Initializes tasks from the provided dataset.

        Args:
            dataset (List[Dict[str, Any]]): The dataset containing task information.
        """
        for i, data in enumerate(dataset):
            data["question_id"] = data["id"]
            if "question_id" not in data:
                data = {"question_id": i, **data}

            if "qd_verdict" not in data:
                data["qd_verdict"] = QDVerdict(
                    correctness="correct",
                    relevance="relevant",
                    completeness="complete",
                    granularity="optimal",
                    coherence="coherent",
                )
            self.update_final_predictions(data["question_id"])
            task = Task(**data)
            task.question_id = data["id"]
            self.tasks.append(task)
        self.total_number_of_tasks = len(self.tasks)
        print(f"Total number of tasks: {self.total_number_of_tasks}")

    def run_tasks(self, results_available: bool = False):
        """Runs the tasks using a pool of workers."""
        print(f"Running tasks with {self.args.num_workers} workers.")
        total_tasks = len(self.tasks)
        if self.args.num_workers > 1:
            with Manager() as manager:
                pbar = tqdm(total=total_tasks)

                # Wrap original callback to update tqdm
                def callback_with_progress(log):
                    self.task_done(log)
                    pbar.update(1)

                with Pool(self.args.num_workers) as pool:
                    for task in self.tasks:
                        pool.apply_async(
                            self.worker, args=(task,), callback=callback_with_progress
                        )
                    pool.close()
                    pool.join()
                pbar.close()
        else:
            for task in tqdm(self.tasks):
                log = self.worker(task, results_available=results_available)
                self.task_done(log, results_available=results_available)

    def worker(self, task: Task, results_available: bool = False) -> Tuple[Any, str]:
        """
        Worker function to process a single task.

        Args:
            task (Task): The task to be processed.

        Returns:
            tuple: The state of the task processing and task identifiers.
        """
        print(f"Initializing task: {task.question_id}")
        # DatabaseManager(db_mode=self.args.data_mode, db_id=task.db_id)
        logger = Logger(
            question_id=task.question_id, result_directory=self.result_directory
        )
        logger._set_log_level(self.args.log_level)

        if not results_available:
            team = build_team(self.args.config)
            # print(f"Team built for task {task.question_id}.")

        thread_id = f"{self.args.run_start_time}_{task.question_id}"
        thread_config = {"configurable": {"thread_id": thread_id}}
        # print(task.model_dump())
        state_values = SystemState(task=task, execution_history=[])
        thread_config["recursion_limit"] = 50
        if not results_available:
            for state_dict in team.stream(
                state_values, thread_config, stream_mode="values"
            ):
                logger.log(
                    "________________________________________________________________________________________"
                )
                continue
            system_state = SystemState(**state_dict)
        else:
            system_state = state_values
        return system_state, task.question_id

    def task_done(self, log: Tuple[SystemState, str], results_available=False) -> None:
        """
        Callback function when a task is done.

        Args:
            log (tuple): The log information of the task processing.
        """
        state, question_id = log
        print(f"Task done for question_id: {question_id}")
        if state is None:
            print("State is None: ", state)
            return
        # print(state.execution_history)
        if state.execution_history == [] and results_available:
            try:
                with open(
                    os.path.join(self.result_directory, f"{question_id}.json"), "r"
                ) as f:
                    state.execution_history = json.load(f)
            except FileNotFoundError:
                return
        for step in state.execution_history:
            # print(step)
            if "qd_result" in step:
                print(f"Updating final result for question_id: {question_id}")
                try:
                    self.statistics_manager.update_stats(
                        question_id, "qd_result", step["qd_result"]
                    )
                except TypeError:
                    print(
                        f"!!!Skipping qid: {question_id}. REASON: `exec_res` is None!!!"
                    )
                    return
                
            if "final_result" in step:
                print(f"Updating final result for question_id: {question_id}")
                try:
                    self.statistics_manager.update_stats(
                        question_id, "final_result", step["final_result"], state
                    )
                except TypeError:
                    print(
                        f"!!!Skipping qid: {question_id}. REASON: `exec_res` is None!!!"
                    )
                    return
                try:
                    self.update_final_predictions(
                        question_id, step["final_result"]["predicted_answer"]
                    )
                except TypeError:
                    print(
                        f"!!!Skipping qid: {question_id}. TypeError: 'NoneType' object is not subscriptable!!!"
                    )
                    return
        self.statistics_manager.dump_statistics_to_file()
        self.processed_tasks += 1
        self.plot_progress()

    def plot_progress(self, bar_length: int = 100):
        """
        Plots the progress of task processing.

        Args:
            bar_length (int, optional): The length of the progress bar. Defaults to 100.
        """
        processed_ratio = self.processed_tasks / self.total_number_of_tasks
        progress_length = int(processed_ratio * bar_length)
        print("\x1b[1A" + "\x1b[2K" + "\x1b[1A")  # Clear previous line
        print(
            f"[{'=' * progress_length}>{' ' * (bar_length - progress_length)}] {self.processed_tasks}/{self.total_number_of_tasks}"
        )

    def generate_sql_files(self):
        """Generates SQL files from the execution history."""
        sqls = {}

        for file in os.listdir(self.result_directory):
            if file.endswith(".json") and "_" in file:
                _index = file.find("_")
                question_id = file.replace(".json", "")
                # db_id = file[_index + 1:-5]
                with open(os.path.join(self.result_directory, file), "r") as f:
                    try:
                        exec_history = json.load(f)
                    except json.JSONDecodeError:
                        print(f"Error decoding JSON from file: {file}")
                        raise json.JSONDecodeError(
                            f"Error decoding JSON from file: {file}"
                        )
                    except FileNotFoundError:
                        print(f"File not found: {file}")
                        raise FileNotFoundError(f"File not found: {file}")
                    except Exception as e:
                        print(f"An error occurred while processing file {file}: {e}")
                        raise e
                    for step in exec_history:
                        if "SQL" in step:
                            tool_name = step["tool_name"]
                            if tool_name not in sqls:
                                sqls[tool_name] = {}
                            sqls[tool_name][question_id] = step["SQL"]
        for key, value in sqls.items():
            with open(os.path.join(self.result_directory, f"-{key}.json"), "w") as f:
                json.dump(value, f, indent=4)
