from collections import defaultdict
import json
import time
import datetime
import torch
from torch_geometric.data import HeteroData


def _data_loader(
    users_file: str = "../data/all_users.json",
    problems_file: str = "../data/all_problems.json",
    holds_file: str = "../data/all_holds.json",
):
    """
    Load users, problems, and problem holds data from JSON files.
    """
    with open(users_file, "r") as f:
        users = json.load(f)

    with open(problems_file, "r") as f:
        problems = json.load(f)

    with open(holds_file, "r") as f:
        problem_holds = json.load(f)

    return users, problems, problem_holds


def data_cleaner(users: dict, problems: dict, problem_holds: dict):
    # Create a dictionary where the unique holds are the keys and
    # the values are the problems associated with each hold
    holds = defaultdict(lambda: {"start": [], "middle": [], "end": []})

    for problem, pholds in problem_holds.items():
        for section in ("start", "middle", "end"):
            for hold in pholds[section]:
                holds[hold][section].append(problem)

    holds = dict(holds)

    # Rename the keys in problems to match those in holds
    problems = {key.replace(" ", "_"): value for key, value in problems.items()}

    # Remove problems that have no holds
    valid_problems = set(problem_holds.keys())
    problems_no_holds = []
    for key in problems.keys():
        if key not in valid_problems:
            problems_no_holds.append(key)

    # Remove users that have no problems or no valid problems
    # and clean up their problems field accordingly
    empty_users = []
    for key, user in users.items():
        if not "problems" in user:
            # Remove the user if they have no problems field
            empty_users.append(key)
        else:
            # Remove all the problems that are not in valid_problems
            user_problems = user["problems"]
            filtered_problems = {
                problem.replace(" ", "_"): value
                for problem, value in user_problems.items()
                if problem.replace(" ", "_") in valid_problems
            }
            # If the user has no problems left, mark them for removal
            if len(filtered_problems) == 0:
                empty_users.append(key)
            else:
                users[key]["problems"] = filtered_problems

    for user_id in empty_users:
        del users[user_id]
    for problem_id in problems_no_holds:
        del problems[problem_id]

    return users, problems, holds


def get_grade_mappings(users: dict, problems: dict):
    # Collect all GRADES observed in users and problems
    grades = set()
    for u in users.values():
        if u["highest_grade"] is not None:
            grades.add(u["highest_grade"])
        for pinfo in u["problems"].values():
            if pinfo["grade"] is not None:
                grades.add(pinfo["grade"])

    for p in problems.values():
        if p["grade"] is not None:
            grades.add(p["grade"])

    grades = sorted(grades)
    grade_to_idx = {g: i for i, g in enumerate(grades)}
    return grade_to_idx


def grade_encoder(g, grade_to_idx):
    # Encode grade g to its integer index, or -1 if g is None
    if g is None:
        return -1
    return grade_to_idx[g]


def users_feature_matrix(users: dict, user_ids: list, grade_to_idx: dict):
    user_features = []

    for uid in user_ids:
        user = users[uid]

        # All current numerical features
        ranking = float(user["ranking"]) if user["ranking"] is not None else 0.0
        highest_grade_idx = float(grade_encoder(user["highest_grade"], grade_to_idx))
        height = float(user["height"]) if user["height"] is not None else 0.0
        weight = float(user["weight"]) if user["weight"] is not None else 0.0
        problems_sent = (
            float(user["problems_sent"]) if user["problems_sent"] is not None else 0.0
        )
        # bio =

        user_features.append(
            [ranking, highest_grade_idx, height, weight, problems_sent]
        )

    user_x = torch.tensor(user_features, dtype=torch.float)
    return user_x


def problems_feature_matrix(problems: dict, problem_ids: list, grade_to_idx: dict):
    problem_features = []

    for pid in problem_ids:
        problem = problems[pid]

        # All current numerical features
        grade_idx = float(grade_encoder(problem["grade"], grade_to_idx))
        rating = float(problem["rating"]) if problem["rating"] is not None else 0.0
        num_sends = (
            float(problem["num_sends"]) if problem["num_sends"] is not None else 0.0
        )
        # setter =
        # holds =

        problem_features.append([grade_idx, rating, num_sends])

    problem_x = torch.tensor(problem_features, dtype=torch.float)
    return problem_x


def user_problem_edge_creation(
    users: dict,
    user_id_to_idx: dict,
    problem_id_to_idx: dict,
    encode_grade,
):
    up_user_indices = []
    up_problem_indices = []
    up_edge_grades = []
    up_edge_ratings = []
    up_edge_dates = []
    up_edge_attempts = []
    # edge_comments = []

    for uid, u in users.items():
        u_idx = user_id_to_idx[uid]
        for prob_name, interaction in u["problems"].items():
            if prob_name not in problem_id_to_idx:
                # Just in case we missed removing some problems
                continue

            p_idx = problem_id_to_idx[prob_name]

            up_user_indices.append(u_idx)
            up_problem_indices.append(p_idx)

            up_edge_grades.append(float(encode_grade(interaction["grade"])))
            up_edge_ratings.append(
                float(interaction["rating"])
                if interaction["rating"] is not None
                else 0.0
            )
            up_edge_attempts.append(
                float(interaction["attempts"])
                if interaction["attempts"] is not None
                else 0.0
            )
            up_edge_dates.append(
                time.mktime(
                    datetime.datetime.strptime(
                        interaction["date"], "%Y-%m-%d"
                    ).timetuple()
                )
            )  # Unix timestamp
            # edge_comments.append(...)

    up_edge_index = torch.tensor(
        [up_user_indices, up_problem_indices], dtype=torch.long
    )
    up_edge_attr = torch.tensor(
        list(
            zip(up_edge_grades, up_edge_ratings, up_edge_attempts)
        ),  # shape: [num_edges, 3]
        dtype=torch.float,
    )
    up_edge_time = torch.tensor(up_edge_dates, dtype=torch.float)

    return up_edge_index, up_edge_attr, up_edge_time


def problem_hold_edge_creation(
    holds: dict,
    hold_id_to_idx: dict,
    problem_id_to_idx: dict,
):
    hp_hold_indices = []
    hp_problem_indices = []
    hp_is_start = []
    hp_is_middle = []
    hp_is_end = []

    for hold, problems in holds.items():
        h_idx = hold_id_to_idx[hold]
        for type in ("start", "middle", "end"):
            for problem in problems[type]:
                p_idx = problem_id_to_idx[problem]

                hp_hold_indices.append(h_idx)
                hp_problem_indices.append(p_idx)

                hp_is_start.append(int(type == "start"))
                hp_is_middle.append(int(type == "middle"))
                hp_is_end.append(int(type == "end"))

    hp_edge_index = torch.tensor(
        [hp_hold_indices, hp_problem_indices], dtype=torch.long
    )
    hp_edge_attr = torch.tensor(
        list(zip(hp_is_start, hp_is_middle, hp_is_end)), dtype=torch.float
    )

    return hp_edge_index, hp_edge_attr


def create_hetero_graph():
    # Load data
    users, problems, problem_holds = _data_loader()

    # Clean data
    users, problems, holds = data_cleaner(users, problems, problem_holds)

    # Integer ids are needed for PyG tensors
    user_ids = list(users.keys())
    problem_ids = list(problems.keys())
    hold_ids = list(holds.keys())

    user_id_to_idx = {uid: i for i, uid in enumerate(user_ids)}
    problem_id_to_idx = {pid: i for i, pid in enumerate(problem_ids)}
    hold_id_to_idx = {hid: i for i, hid in enumerate(hold_ids)}

    hetero_data = HeteroData()

    # Get grade mappings
    grade_to_idx = get_grade_mappings(users, problems)
    encode_grade = lambda g: grade_encoder(g, grade_to_idx)

    # Create feature matrices
    user_x = users_feature_matrix(users, user_ids, grade_to_idx)
    problem_x = problems_feature_matrix(problems, problem_ids, grade_to_idx)
    hold_x = torch.eye(len(hold_ids))  # One-hot encoding for holds

    # Create edges
    up_edge_index, up_edge_attr, up_edge_time = user_problem_edge_creation(
        users, user_id_to_idx, problem_id_to_idx, encode_grade
    )
    hp_edge_index, hp_edge_attr = problem_hold_edge_creation(
        holds, hold_id_to_idx, problem_id_to_idx
    )

    # Construct the heterogeneous graph

    # Add the nodes and their features
    hetero_data["user"].x = user_x  # [num_users, user_feat_dim]
    hetero_data["problem"].x = problem_x  # [num_problems, problem_feat_dim]
    hetero_data["hold"].x = hold_x  # One-hot encoding for holds

    # Add edges between users and problems
    hetero_data["user", "rates", "problem"].edge_index = up_edge_index  # [2, num_edges]
    hetero_data["user", "rates", "problem"].edge_attr = (
        up_edge_attr  # [num_edges, edge_feat_dim]
    )
    hetero_data["user", "rates", "problem"].edge_time = up_edge_time  # [num_edges,]

    # Add reverse edges (apparently good for GNN message passing):
    hetero_data["problem", "rated_by", "user"].edge_index = up_edge_index.flip(0)
    hetero_data["problem", "rated_by", "user"].edge_attr = (
        up_edge_attr  # usually same attrs
    )
    hetero_data["problem", "rated_by", "user"].edge_time = up_edge_time

    # Add edges between problems and holds
    hetero_data["problem", "contains", "hold"].edge_index = hp_edge_index
    hetero_data["problem", "contains", "hold"].edge_attr = hp_edge_attr

    # Add reverse edges
    hetero_data["hold", "contained_in", "problem"].edge_index = hp_edge_index.flip(0)
    hetero_data["hold", "contained_in", "problem"].edge_attr = hp_edge_attr

    return hetero_data
