import argparse
import yaml
import json
import os
from datetime import datetime
from typing import Any, Dict, List
from pathlib import Path

from runner.run_manager import RunManager

def parse_arguments() -> argparse.Namespace:
    """
    Parses command-line arguments.

    Returns:
        argparse.Namespace: The parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Run the pipeline with the specified configuration.")
    parser.add_argument('--data_mode', type=str, required=True, choices=['train', 'dev'], help="Mode of the data to be processed.")
    parser.add_argument('--data_path', type=str, required=True, help="Path to the data file.")
    # parser.add_argument('--oracle_path', type=str, required=True, help="Path to the oracle file.")
    parser.add_argument('--config', type=str, required=True, help="Path to the configuration file.")
    parser.add_argument('--num_workers', type=int, default=1, help="Number of workers to use.")
    parser.add_argument('--log_level', type=str, default='warning', help="Logging level.")
    parser.add_argument('--result_directory', type=str, default=None, help="Custom result directory.")
    parser.add_argument('--pick_final_sql', type=bool, default=False, help="Pick the final SQL from the generated SQLs.")
    parser.add_argument('--start', type=int, default=0, help='Starting index for processing (default: 0)')
    parser.add_argument('--end', type=int, default=9999, help='Ending index for processing (default: 9999)')
    parser.add_argument('--save_dir', type=str, default='',  help="Directory to save/load results")
    args = parser.parse_args()

    args.run_start_time = datetime.now().isoformat()
    with open(args.config, 'r') as file:
        args.config=yaml.safe_load(file)
    
    if(args.save_dir):
        RESULT_ROOT_PATH = "results"
        data_mode = args.data_mode
        setting_name = args.config["setting_name"]
        dataset_name = Path(args.data_path).stem
        run_folder_name = str(args.save_dir)
        run_folder_path = Path(RESULT_ROOT_PATH) / data_mode / setting_name / dataset_name / run_folder_name
        final_args_file = run_folder_path / "-args.json"
        if(final_args_file.exists()):
            return argparse.Namespace(**json.load(open(final_args_file, "r")))
        else:
            pass
        
    return args

def load_dataset(args) -> List[Dict[str, Any]]:
    """
    Loads the dataset from the specified path.

    Args:
        data_path (str): Path to the data file.

    Returns:
        List[Dict[str, Any]]: The loaded dataset.
    """
    with open(args.data_path, 'r') as file:
        dataset = [json.loads(line) for line in file if line.strip()]

    RESULT_ROOT_PATH = "results"
    data_mode = args.data_mode
    setting_name = args.config["setting_name"]
    dataset_name = Path(args.data_path).stem
    run_folder_name = str(args.run_start_time)
    run_folder_path = Path(RESULT_ROOT_PATH) / data_mode / setting_name / dataset_name / run_folder_name
    final_prediction_file = run_folder_path / "-predictions.json"
    exists_question_id = []
    if final_prediction_file.exists():
        with open(final_prediction_file, 'r') as f:
            results = json.load(f)
            for key, value in results.items():
                if(value!=0):
                    exists_question_id.append(int(key))

    small_dataset = dataset[max(0,args.start):min(len(dataset), args.end)]
    print(f"Total number of questions: {len(small_dataset)}")
    print(f"Number of questions that have been pruned: {len(exists_question_id)}")

    new_small_dataset = []
    for i, data in enumerate(small_dataset,start=max(0,args.start)):
        if "question_id" not in data:
            data = {"question_id": i, **data}
        if data['question_id'] in exists_question_id:
            continue
        new_small_dataset.append(data)

    # print(f"Total number of tasks: {len(new_small_dataset)}")
    return new_small_dataset

def main():
    """
    Main function to run the pipeline with the specified configuration.
    """
    args = parse_arguments()

    dataset = load_dataset(args)

    print('Starting from',args.start,' to ',args.end)

    run_manager = RunManager(args)
    run_manager.initialize_tasks(dataset)
    if args.result_directory:
        run_manager.run_tasks(results_available=True)
    else:
        run_manager.run_tasks()
    run_manager.generate_sql_files()

if __name__ == '__main__':
    main()
