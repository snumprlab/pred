import os
DATA_FOLDER=os.environ['DATA_DIR']
import json
from pathlib import Path
from itertools import chain
#all_train_edh_files = list(
#    (Path(DATA_FOLDER)/"edh_instances/train").glob("*.json")
#)
#all_valid_edh_files = list(chain(
#    (Path(DATA_FOLDER)/"edh_instances/valid_seen").glob("*.json"),
#    (Path(DATA_FOLDER)/"edh_instances/valid_unseen").glob("*.json")
#))

from typing import List
def _get_dialog_history(edh_instance: dict) -> List[str]:
    return [
        f"{speaker}: {utterance}\n"
            for speaker, utterance in edh_instance["dialog_history"]
    ]
# test
# _get_dialog_history(json.load(all_train_edh_files[0].open("r")))


from typing import Dict, Union
from functools import lru_cache
def _get_task_type_and_params(edh_instance: dict) -> Dict[str, Union[str, int]]:
    @lru_cache()
    def _internal(game_id: str) -> Dict[str, Union[str, int]]:
        game = json.load((Path(DATA_FOLDER)/"all_game_files"/f"{game_id}.game.json").open("r"))
        task_type = game["tasks"][0]["task_name"]  # assuming each game has only one task
        if not "X" in task_type.split(" "): # hacky
            # no param for this task
            obj_count, obj_target, parent_target = None, None, None
        elif not "Y" in task_type.split(" "): # hacky as well
            # 1 param for this task
            obj_count, obj_target = None, game["tasks"][0]["task_params"][0]
            parent_target = None
        elif not "N" in task_type.split(" "): # hacky as well
            # 2 params for this task
            obj_count, obj_target, parent_target = None, game["tasks"][0]["task_params"][0], game["tasks"][0]["task_params"][2]
        else:
            # 3 params for this task
            obj_count, obj_target, parent_target = game["tasks"][0]["task_params"][0], game["tasks"][0]["task_params"][1], game["tasks"][0]["task_params"][3]
        return dict(task_type=task_type, obj_count=obj_count, obj_target=obj_target, parent_target=parent_target)
    game_id = edh_instance["game_id"]
    return _internal(game_id)
