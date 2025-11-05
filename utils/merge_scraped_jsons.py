import os
import json

data = os.listdir("./data")
users = [x for x in data if x.startswith("users_")]
problems = [x for x in data if x.startswith("problems_")]
users.sort(key=lambda x: int(x.split("_")[1]))
problems.sort(key=lambda x: int(x.split("_")[1]))

for user in users:
    with open(f"./data/{user}", "r", encoding="utf-8") as f:
        user_data = json.load(f)
    if user == users[0]:
        all_users = user_data
    else:
        all_users.update(user_data)


for problem in problems:
    with open(f"./data/{problem}", "r", encoding="utf-8") as f:
        problem_data = json.load(f)
    if problem == problems[0]:
        all_problems = problem_data
    else:
        for name, info in problem_data.items():
            if name not in all_problems:
                all_problems[name] = info
            else:
                all_problems[name]["num_sends"] += 1

with open("./data/all_users.json", "w", encoding="utf-8") as f:
    json.dump(all_users, f, indent=4, ensure_ascii=False)

with open("./data/all_problems.json", "w", encoding="utf-8") as f:
    json.dump(all_problems, f, indent=4, ensure_ascii=False)
