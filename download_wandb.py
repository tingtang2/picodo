"""Downloads all logged runs from WandB and stores them into a dataframe."""

from __future__ import annotations

import fire
import pandas as pd
import wandb
from tqdm import tqdm
import time

WANDB_PROJECT = "loss-spikes-edu"
WANDB_ENTITY = "sophie-qx-li-ucla"


def download_results(
    file: str | None = None,
    group: str | None = None,
    created_after: str | None = None,
    keys: list[str] | None = None,
) -> pd.DataFrame:
    """Download results from Weights and Biases.

    :param file: File name.
    :param group: WandB group of the experiment.
    :param created_after: Only download runs created after the specified date, e.g.'2024-01-01T##'.
    :param keys: Which columns of the run table to return. Beware, only the join of table columns is returned.
    """
    api = wandb.Api(timeout=500)
    print(keys)

    # Project is specified by <entity/project-name>
    filters = {}
    if group is not None:
        filters["group"] = group
    if created_after is not None:
        filters["$and"] = [
            {"created_at": {"$lt": "2099-01-01T##", "$gt": created_after}}
        ]
    runs = api.runs(
        WANDB_ENTITY + "/" + WANDB_PROJECT,
        filters=filters,
        # per_page=1000,
    )

    all_runs_df_list = []

    for run in tqdm(runs):

        tries = 0
        max_retries = 5
        while tries < max_retries:
            # Logged values for all steps from this run
            try:
                '''
                run_df = pd.DataFrame(
                    #run.beta_scan_history(keys=keys),
                    run.beta_scan_history(),
                    # page_size=100,
                )
                '''
                history_gen = run.scan_history(keys=["_step", "train_loss", "train_output_logit_norm"])
                run_df = pd.DataFrame(list(history_gen))
                # run_df = run.history()
                break
            except Exception as e:
                tries += 1
                print(f"retrying: {tries}: {e}")
                if tries == max_retries:
                    print("Maximum retries reached. Raising exception.")
                    raise e
                time.sleep(2)  # Wait before the next attempt

        run_df.insert(0, "run name", run.name)
        run_df.insert(1, "experiment", run.group)
        run_df.insert(
            2, "gpu", None if run.metadata is None else run.metadata.get("gpu", None)
        )
        try:
            runtime_s = run.summary["_runtime"]
        except KeyError:
            # If the run is not finished, we cannot get the runtime.
            runtime_s = None
        run_df.insert(3, "Runtime (s)", runtime_s)

        # run.config is the input metrics.
        # We remove special values that start with _.
        config = {k: v for k, v in run.config.items() if not k.startswith("_")}
        for idx, (k, v) in enumerate(config.items()):
            if isinstance(v, list):
                v = str(v)
            run_df.insert(idx + 3, k, v)

        all_runs_df_list.append(run_df)

    all_runs_df = pd.concat(all_runs_df_list)
    if file is not None:
        all_runs_df.to_csv(file)

    return all_runs_df


if __name__ == "__main__":
    fire.Fire(download_results)
